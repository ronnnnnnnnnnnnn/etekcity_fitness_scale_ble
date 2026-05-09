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
  - ``birthdate`` / ``height`` / ``sex`` — user PII

What we keep verbatim:
  - User IDs (the empty string is a structural marker for the v1 legacy
    default user, not PII; UUIDs are opaque)
  - Timestamps, weight, impedance, body composition values — useful for
    debugging, not PII in the same way an address is
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_BIRTHDATE,
    CONF_HEIGHT,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_PERSON_ENTITY,
    CONF_SEX,
    CONF_USER_PROFILES,
    DOMAIN,
)
from .coordinator import ScaleDataUpdateCoordinator

_TO_REDACT = {
    "address",
    "unique_id",
    CONF_PERSON_ENTITY,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_BIRTHDATE,
    CONF_HEIGHT,
    CONF_SEX,
}


def _redact_user_profiles(profiles: list[dict]) -> list[dict]:
    """Redact PII fields inside each user profile."""
    return [async_redact_data(p, _TO_REDACT) for p in profiles]


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for an Etekcity scale config entry."""
    data = dict(entry.data)
    if CONF_USER_PROFILES in data:
        data[CONF_USER_PROFILES] = _redact_user_profiles(data[CONF_USER_PROFILES])

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
        user_history[user_id] = {
            "history_count": len(history),
            "latest_timestamp": latest.get("timestamp") if latest else None,
            "latest_weight_kg": latest.get("weight_kg") if latest else None,
            "latest_has_impedance": (
                latest.get("impedance_ohm") is not None if latest else False
            ),
        }

    # Pending measurements are keyed by ISO timestamp. Drop notified_mobile_services
    # (PII via service names) and any non-JSON-safe raw payloads, keep the rest.
    pending_redacted: dict[str, dict[str, Any]] = {}
    for ts, info in coordinator.get_pending_measurements().items():
        pending_redacted[ts] = {
            k: v
            for k, v in info.items()
            if k not in {"raw_measurement", "notified_mobile_services"}
        }

    client = getattr(coordinator, "_client", None)
    diag["coordinator"] = {
        "device_name": coordinator._device_name,
        "scale_model": str(coordinator._scale_model),
        "display_unit": coordinator.get_display_unit().name,
        "user_count": len(profiles),
        "user_ids": [p.get("user_id", "") for p in profiles],
        "user_has_body_metrics": {
            p.get("user_id", ""): bool(p.get("body_metrics_enabled", False))
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
