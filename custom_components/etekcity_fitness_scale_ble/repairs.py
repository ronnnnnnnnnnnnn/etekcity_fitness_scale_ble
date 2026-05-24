"""Repair-issue scanning for etekcity_fitness_scale_ble.

Surfaces drift between the integration's stored config and the
surrounding Home Assistant state on the Settings → Repairs page.

Note on the location filter: at runtime, the location filter only
*excludes* a candidate when their linked person entity is in state
exactly ``"not_home"``. Missing entities, ``unknown``, ``unavailable``,
and any other state are all kept as candidates (and there's a safety
net: if filtering empties the list, the unfiltered list is used). So
the broken-link issues below don't actively drop measurements — they
mean the location signal the user *configured* isn't doing anything,
and the integration falls back to weight-only matching for that user.

``mobile_service_missing`` and ``person_entity_missing`` are fixable
in-place — the right one-click action is unambiguous (drop the stale
entry / clear the broken link). ``person_entity_unknown`` is left
non-fixable because the right remediation usually isn't clearing the
link, it's assigning a tracker to the existing person; that's a
decision the user has to make.

Issue types:

- ``person_entity_missing_<entry_id>_<user_id>`` *(fixable)*
    The user's profile links to a ``person.X`` entity that no longer
    exists. The link is silently ignored at runtime (a missing entity
    doesn't exclude the user from auto-assignment), so the practical
    impact is just that the configured location signal is dead weight.
    The fix flow clears the link from the profile; relinking via
    Configure remains available for users who want to point at a new
    entity instead.

- ``person_entity_unknown_<entry_id>_<user_id>`` *(non-fixable)*
    The linked ``person.X`` exists but its state is ``unknown`` — the
    person was created in HA but no device trackers have been assigned
    to it. Same runtime impact as the missing case (not excluded, just
    no useful location signal). Surfaced as a warning so the user knows
    their configuration is degraded. Left non-fixable because the right
    remediation is usually "assign a tracker", not "clear the link".
    ``unavailable`` is deliberately *not* flagged because it's typically
    transient (HA restart, brief tracker outage); raising a flapping
    issue every reload would just be noise.

- ``mobile_service_missing_<entry_id>_<user_id>_<service_slug>`` *(fixable)*
    A configured ``notify.mobile_app_*`` service no longer exists.
    Sending an actionable notification to it would no-op silently —
    typical cause is the companion app being uninstalled or the phone
    renamed. The fix flow removes the stale entry from the profile and
    reloads the entry so the scan re-runs and the issue clears.

The scan runs at ``async_setup_entry`` and is re-scanned on options-flow
reload (which goes through ``async_setup_entry`` again, so no separate
update listener is needed).
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components.repairs import ConfirmRepairFlow, RepairsFlow
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
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

_REPAIR_KIND_MOBILE_SERVICE_MISSING = "mobile_service_missing"
_REPAIR_KIND_PERSON_ENTITY_MISSING = "person_entity_missing"


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

        # Linked person entity in a degraded state. Neither case actively
        # breaks auto-assignment (the location filter only excludes on exact
        # "not_home"), but the configured link isn't providing the location
        # signal the user expected.
        person_entity = profile.get(CONF_PERSON_ENTITY)
        if person_entity:
            person_state = hass.states.get(person_entity)
            if person_state is None:
                # Entity gone (deleted in Settings → People). Fixable:
                # clearing the dead link is unambiguous (relinking via
                # Configure remains an option for users who want that).
                issue_id = _person_entity_missing_issue_id(entry.entry_id, user_id)
                desired[issue_id] = {
                    "translation_key": "person_entity_missing",
                    "translation_placeholders": {
                        "user_name": user_name,
                        "person_entity": person_entity,
                    },
                    "severity": ir.IssueSeverity.WARNING,
                    "is_fixable": True,
                    "data": {
                        "kind": _REPAIR_KIND_PERSON_ENTITY_MISSING,
                        "entry_id": entry.entry_id,
                        "user_id": user_id,
                    },
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

        # Mobile notify services that no longer exist. Fixable in-place:
        # the only meaningful remediation is to drop the stale entry, so
        # expose a one-click confirm flow (see ``async_create_fix_flow``).
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
                    "is_fixable": True,
                    "data": {
                        "kind": _REPAIR_KIND_MOBILE_SERVICE_MISSING,
                        "entry_id": entry.entry_id,
                        "user_id": user_id,
                        "service": service,
                    },
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
            is_fixable=kwargs.pop("is_fixable", False),
            data=kwargs.pop("data", None),
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


# ---------------------------------------------------------------------------
# Fix flows. Each handles a single repair "kind" and follows the same shape:
# show a confirm step → mutate the user's profile in entry.data → reload the
# entry so the next scan clears the now-resolved issue.
# ---------------------------------------------------------------------------


class _ClearPersonEntityRepairFlow(RepairsFlow):
    """Clear a broken person_entity link from a user profile.

    Called from the Repairs UI when the user clicks **Submit** on a
    ``person_entity_missing`` issue. Clearing is unambiguous; users who
    want to point at a different person entity instead can do that via
    Configure → Edit User.
    """

    def __init__(self, entry_id: str, user_id: str) -> None:
        self._entry_id = entry_id
        self._user_id = user_id

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is None:
            return self.async_show_form(step_id="confirm", data_schema=vol.Schema({}))

        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if entry is None:
            return self.async_create_entry(title="", data={})

        profiles = list(entry.data.get(CONF_USER_PROFILES, []))
        changed = False
        for i, profile in enumerate(profiles):
            if profile.get(CONF_USER_ID) != self._user_id:
                continue
            if profile.get(CONF_PERSON_ENTITY):
                # Drop the key entirely; the rest of the integration treats
                # missing-key and explicit-None the same, but keeping the
                # stored shape minimal is friendlier when inspecting the
                # config entry by hand.
                profiles[i] = {
                    k: v for k, v in profile.items() if k != CONF_PERSON_ENTITY
                }
                changed = True
            break

        if changed:
            self.hass.config_entries.async_update_entry(
                entry, data={**entry.data, CONF_USER_PROFILES: profiles}
            )
            await self.hass.config_entries.async_reload(self._entry_id)

        return self.async_create_entry(title="", data={})


class _RemoveMobileServiceRepairFlow(RepairsFlow):
    """Drop a no-longer-registered notify service from a user profile.

    Called from the Repairs UI when the user clicks **Submit** on a
    ``mobile_service_missing`` issue. The flow is a single confirm step;
    on confirm we mutate the entry's stored profiles and reload, which
    re-runs ``async_scan_repair_issues`` and clears the now-resolved issue.
    """

    def __init__(self, entry_id: str, user_id: str, service: str) -> None:
        self._entry_id = entry_id
        self._user_id = user_id
        self._service = service

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is None:
            return self.async_show_form(step_id="confirm", data_schema=vol.Schema({}))

        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if entry is None:
            # Entry vanished between issue creation and the user clicking
            # Submit (e.g. integration removed). Nothing to update — the
            # issue will be swept on the next scan / clear.
            return self.async_create_entry(title="", data={})

        profiles = list(entry.data.get(CONF_USER_PROFILES, []))
        changed = False
        for i, profile in enumerate(profiles):
            if profile.get(CONF_USER_ID) != self._user_id:
                continue
            services = list(profile.get(CONF_MOBILE_NOTIFY_SERVICES) or [])
            if self._service in services:
                services.remove(self._service)
                profiles[i] = {
                    **profile,
                    CONF_MOBILE_NOTIFY_SERVICES: services,
                }
                changed = True
            break

        if changed:
            self.hass.config_entries.async_update_entry(
                entry, data={**entry.data, CONF_USER_PROFILES: profiles}
            )
            # Reload so async_setup_entry re-runs the repair scan and the
            # now-resolved issue is deleted from the registry.
            await self.hass.config_entries.async_reload(self._entry_id)

        return self.async_create_entry(title="", data={})


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Home Assistant entry point for the Repairs fix-flow dispatcher.

    HA discovers this function on the integration's ``repairs`` module
    and calls it when the user clicks **Fix Issue** on one of our
    fixable issues. Non-fixable issues never reach this function.
    """
    if (
        data
        and isinstance(data.get("entry_id"), str)
        and isinstance(data.get("user_id"), str)
    ):
        kind = data.get("kind")
        if kind == _REPAIR_KIND_MOBILE_SERVICE_MISSING and isinstance(
            data.get("service"), str
        ):
            return _RemoveMobileServiceRepairFlow(
                entry_id=data["entry_id"],
                user_id=data["user_id"],
                service=data["service"],
            )
        if kind == _REPAIR_KIND_PERSON_ENTITY_MISSING:
            return _ClearPersonEntityRepairFlow(
                entry_id=data["entry_id"],
                user_id=data["user_id"],
            )
    # Fallback for any future fixable issue we haven't wired up yet — a
    # plain confirm-and-dismiss flow is safer than raising on unknown data.
    return ConfirmRepairFlow()
