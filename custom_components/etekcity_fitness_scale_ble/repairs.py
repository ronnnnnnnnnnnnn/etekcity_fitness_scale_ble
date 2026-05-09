"""Repair-issue scanning for etekcity_fitness_scale_ble.

Surfaces drift between the integration's stored config and the
surrounding Home Assistant state on the Settings → Repairs page.
Issues are non-fixable from the Repairs UI itself — the fix is to open
the user's profile in the integration's options flow and correct the
field. Each issue's translated text tells the user what to do.

Issue types:

- ``person_entity_missing_<entry_id>_<user_id>``
    The user's profile points at a ``person.X`` entity that no longer
    exists. The location filter quietly drops users whose person state
    is missing or ``not_home``; once the entity is gone, the user is
    effectively excluded forever, which is rarely what was intended.

- ``person_entity_unknown_<entry_id>_<user_id>``
    The linked ``person.X`` exists but its state is ``unknown`` — the
    person was created in HA but no device trackers have been assigned
    to it. The location filter treats this as "not home", so the user
    is permanently excluded from auto-assignment. ``unavailable`` is
    deliberately *not* flagged because it's typically transient (HA
    restart, brief tracker outage); raising a flapping issue every
    reload would just be noise.

- ``mobile_service_missing_<entry_id>_<user_id>_<service_slug>``
    A configured ``notify.mobile_app_*`` service no longer exists.
    Sending an actionable notification to it would no-op silently —
    typical cause is the companion app being uninstalled or the phone
    renamed.

The scan runs at ``async_setup_entry`` and is re-scanned on options-flow
reload (which goes through ``async_setup_entry`` again, so no separate
update listener is needed).
"""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from .const import (
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_PERSON_ENTITY,
    CONF_USER_ID,
    CONF_USER_NAME,
    CONF_USER_PROFILES,
    DOMAIN,
    parse_notify_service,
)


def _person_entity_missing_issue_id(entry_id: str, user_id: str) -> str:
    return f"person_entity_missing_{entry_id}_{user_id}"


def _person_entity_unknown_issue_id(entry_id: str, user_id: str) -> str:
    return f"person_entity_unknown_{entry_id}_{user_id}"


def _mobile_service_issue_id(entry_id: str, user_id: str, service: str) -> str:
    # ``service`` may contain ``.``; replace so the issue_id is filesystem-
    # and registry-safe.
    safe = service.replace(".", "_")
    return f"mobile_service_missing_{entry_id}_{user_id}_{safe}"


def _notify_service_exists(hass: HomeAssistant, service: str) -> bool:
    """Check whether a stored notify-service value resolves to a real
    registered service. Both the short and full forms are accepted via
    ``parse_notify_service`` — see that helper for rationale.
    """
    domain, name = parse_notify_service(service)
    return hass.services.has_service(domain, name)


def async_scan_repair_issues(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Scan the entry's user profiles and reconcile repair issues.

    Creates issues for newly-detected misconfigurations and deletes any
    previously-raised issues that no longer apply (so fixing a profile
    in the options flow clears the issue on the next reload).
    """
    profiles: list[dict[str, Any]] = entry.data.get(CONF_USER_PROFILES, [])

    desired: dict[str, dict[str, Any]] = {}

    for profile in profiles:
        user_id = profile.get(CONF_USER_ID, "")
        user_name = profile.get(CONF_USER_NAME, user_id)
        if not user_id:
            continue

        person_entity = profile.get(CONF_PERSON_ENTITY)
        if person_entity:
            person_state = hass.states.get(person_entity)
            if person_state is None:
                issue_id = _person_entity_missing_issue_id(entry.entry_id, user_id)
                desired[issue_id] = {
                    "translation_key": "person_entity_missing",
                    "translation_placeholders": {
                        "user_name": user_name,
                        "person_entity": person_entity,
                    },
                    "severity": ir.IssueSeverity.WARNING,
                }
            elif person_state.state == STATE_UNKNOWN:
                issue_id = _person_entity_unknown_issue_id(entry.entry_id, user_id)
                desired[issue_id] = {
                    "translation_key": "person_entity_unknown",
                    "translation_placeholders": {
                        "user_name": user_name,
                        "person_entity": person_entity,
                    },
                    "severity": ir.IssueSeverity.WARNING,
                }

        services = profile.get(CONF_MOBILE_NOTIFY_SERVICES) or []
        for service in services:
            if not _notify_service_exists(hass, service):
                issue_id = _mobile_service_issue_id(entry.entry_id, user_id, service)
                desired[issue_id] = {
                    "translation_key": "mobile_service_missing",
                    "translation_placeholders": {
                        "user_name": user_name,
                        "service": service,
                    },
                    "severity": ir.IssueSeverity.WARNING,
                }

    registry = ir.async_get(hass)
    suffix = f"_{entry.entry_id}_"
    existing_for_entry = [
        issue.issue_id
        for (domain, issue_id), issue in registry.issues.items()
        if domain == DOMAIN and suffix in issue_id
    ]
    for issue_id in existing_for_entry:
        if issue_id not in desired:
            ir.async_delete_issue(hass, DOMAIN, issue_id)

    for issue_id, kwargs in desired.items():
        ir.async_create_issue(
            hass,
            DOMAIN,
            issue_id,
            is_fixable=False,
            **kwargs,
        )


def async_clear_repair_issues_for_entry(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Remove all repair issues raised for a given config entry.

    Called from ``async_unload_entry`` so removing the integration
    cleans up its issues from the Repairs page.
    """
    registry = ir.async_get(hass)
    suffix = f"_{entry.entry_id}_"
    for (domain, issue_id), _ in list(registry.issues.items()):
        if domain == DOMAIN and suffix in issue_id:
            ir.async_delete_issue(hass, DOMAIN, issue_id)
