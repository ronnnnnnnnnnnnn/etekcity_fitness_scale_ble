"""Diagnostics support for etekcity_fitness_scale_ble.

Wired up via HA's built-in "Download Diagnostics" button on the device
card. Returns a single JSON blob with the config entry + coordinator
state, with PII redacted.

Redacted fields:
  - ``address``                — BLE MAC of the scale (privacy)
  - ``unique_id``              — entry's BLE MAC
  - ``person_entity``          — links to a HA person entity (privacy)
  - ``mobile_notify_services`` — companion-app service names usually contain
                                 the user's phone hostname
  - ``name``                   — the user's display name (only inside user
                                 profiles: the scale's advertised BLE name
                                 and the device name are model triage data
                                 and are emitted outside the redacted block)
  - ``birthdate`` / ``height`` / ``sex`` — user PII
  - ``user_id``                — a slug *derived from the display name*
                                 ("Alice" → ``alice``), so it is replaced by
                                 a positional alias (``user_1``, ``user_2``…)
                                 that is consistent across the whole dump.
                                 The empty-string id of the v1 legacy default
                                 user is a structural marker, not a name, and
                                 is kept as-is

What we keep verbatim:
  - The last advertisement HA saw from the scale (name, manufacturer data,
    service UUIDs, RSSI, source): it is what identifies a model or hardware
    revision, and asking reporters to dig it out of the advertisement
    monitor by hand cost a round trip on integration issue #44. The QN
    manufacturer payload echoes the scale's MAC, which the entry title
    already carries.
  - Timestamps, weight, impedance, body composition values — useful for
    debugging, not PII in the same way an address is
"""

from __future__ import annotations

from typing import Any

from homeassistant.components import bluetooth
from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_BIRTHDATE,
    CONF_HEIGHT,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_PERSON_ENTITY,
    CONF_SEX,
    CONF_USER_ID,
    CONF_USER_NAME,
    CONF_USER_PROFILES,
    DOMAIN,
)
from .coordinator import ScaleDataUpdateCoordinator

# Applied to ``entry.data`` (and the user profiles inside it) only — never to
# the coordinator block, whose ``last_advertisement.name`` / ``device_name``
# are the scale's, not a person's.
_TO_REDACT = {
    "address",
    "unique_id",
    CONF_USER_NAME,
    CONF_PERSON_ENTITY,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_BIRTHDATE,
    CONF_HEIGHT,
    CONF_SEX,
}


class _UserAliases:
    """Positional ``user_N`` aliases for user ids, stable within one dump.

    Ids are handed out in first-seen order — entry data first, so the
    entry's profiles, the coordinator summaries and the pending candidates
    all agree. The v1 legacy id ``""`` is returned unchanged.
    """

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}

    def __call__(self, user_id: Any) -> Any:
        if not isinstance(user_id, str) or user_id == "":
            return user_id
        return self._aliases.setdefault(user_id, f"user_{len(self._aliases) + 1}")


def _redact_user_profiles(profiles: list[dict], alias: _UserAliases) -> list[dict]:
    """Redact PII fields inside each user profile and alias its id."""
    redacted = []
    for profile in profiles:
        if not isinstance(profile, dict):
            redacted.append(profile)
            continue
        profile = async_redact_data(profile, _TO_REDACT)
        if CONF_USER_ID in profile:
            profile[CONF_USER_ID] = alias(profile[CONF_USER_ID])
        redacted.append(profile)
    return redacted


def _last_advertisement(hass: HomeAssistant, address: str) -> dict[str, Any] | None:
    """The last advertisement HA saw from the scale, JSON-safe, or None."""
    info = bluetooth.async_last_service_info(hass, address, connectable=False)
    if info is None:
        return None
    return {
        "name": info.name,
        "rssi": info.rssi,
        "manufacturer_data": {
            str(company): data.hex() for company, data in info.manufacturer_data.items()
        },
        "service_data": {uuid: data.hex() for uuid, data in info.service_data.items()},
        "service_uuids": list(info.service_uuids),
        "source": info.source,
        "connectable": info.connectable,
        "time": info.time,
    }


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for an Etekcity scale config entry."""
    alias = _UserAliases()
    data = dict(entry.data)
    if isinstance(data.get(CONF_USER_PROFILES), list):
        data[CONF_USER_PROFILES] = _redact_user_profiles(
            data[CONF_USER_PROFILES], alias
        )

    coordinator: ScaleDataUpdateCoordinator | None = hass.data.get(DOMAIN, {}).get(
        entry.entry_id
    )

    diag: dict[str, Any] = {
        "entry": {
            "version": entry.version,
            "title": entry.title,
            "source": entry.source,
            "data": async_redact_data(data, _TO_REDACT),
        },
    }

    if coordinator is None:
        diag["coordinator"] = None
        return diag

    profiles = coordinator.get_user_profiles()

    user_history: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        user_id = profile.get("user_id", "")
        history = coordinator.get_user_history(user_id)
        latest = history[-1] if history else None
        user_history[alias(user_id)] = {
            "history_count": len(history),
            "latest_timestamp": latest.get("timestamp") if latest else None,
            "latest_weight_kg": latest.get("weight_kg") if latest else None,
            "latest_has_impedance": (
                latest.get("impedance_ohm") is not None if latest else False
            ),
        }

    # Pending measurements are keyed by ISO timestamp. Drop notified_mobile_services
    # (PII via service names) and any non-JSON-safe raw payloads, alias the
    # candidate user ids, keep the rest.
    pending_redacted: dict[str, dict[str, Any]] = {}
    for ts, info in coordinator.get_pending_measurements().items():
        pending_redacted[ts] = {
            k: v
            for k, v in info.items()
            if k not in {"raw_measurement", "notified_mobile_services"}
        }
        candidates = pending_redacted[ts].get("candidates")
        if isinstance(candidates, list):
            pending_redacted[ts]["candidates"] = [alias(c) for c in candidates]

    client = getattr(coordinator, "_client", None)
    diag["coordinator"] = {
        "last_advertisement": _last_advertisement(hass, coordinator.address),
        "device_name": coordinator._device_name,
        "scale_model": str(coordinator._scale_model),
        "display_unit": coordinator.get_display_unit().name,
        "user_count": len(profiles),
        "user_ids": [alias(p.get("user_id", "")) for p in profiles],
        "user_has_body_metrics": {
            alias(p.get("user_id", "")): bool(p.get("body_metrics_enabled", False))
            for p in profiles
        },
        "user_history": user_history,
        "pending_count": len(pending_redacted),
        "pending": pending_redacted,
        "scale_connected": client is not None,
        "scale_firmware_revision": (
            getattr(client, "firmware_revision", None) if client else None
        ),
        "scale_battery_level": (
            getattr(client, "battery_level", None) if client else None
        ),
    }
    return diag
