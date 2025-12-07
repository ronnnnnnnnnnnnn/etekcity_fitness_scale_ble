"""The etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime

import voluptuous as vol

from bleak_retry_connector import close_stale_connections_by_address
from homeassistant.components import bluetooth
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_UNIT_SYSTEM, Platform, UnitOfMass
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from urllib.parse import unquote

from .const import (
    CONF_BIRTHDATE,
    CONF_BODY_METRICS_ENABLED,
    CONF_CALC_BODY_METRICS,
    CONF_CREATED_AT,
    CONF_HEIGHT,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_PERSON_ENTITY,
    CONF_SCALE_DISPLAY_UNIT,
    CONF_SEX,
    CONF_UPDATED_AT,
    CONF_USER_ID,
    CONF_USER_NAME,
    CONF_USER_PROFILES,
    CONF_WEIGHT_HISTORY,
    DOMAIN,
)
from .coordinator import ScaleDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR]

# Data keys
DATA_MOBILE_APP_LISTENER_UNSUB = "mobile_app_listener_unsub"

# Service constants
SERVICE_ASSIGN_MEASUREMENT = "assign_measurement"
SERVICE_REASSIGN_MEASUREMENT = "reassign_measurement"
SERVICE_REMOVE_MEASUREMENT = "remove_measurement"
ATTR_TIMESTAMP = "timestamp"
ATTR_USER_ID = "user_id"
ATTR_FROM_USER_ID = "from_user_id"
ATTR_TO_USER_ID = "to_user_id"

# Service schemas
SERVICE_ASSIGN_MEASUREMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_TIMESTAMP): cv.string,
        vol.Required(ATTR_USER_ID): cv.string,
    },
    extra=vol.ALLOW_EXTRA,
)

SERVICE_REASSIGN_MEASUREMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_FROM_USER_ID): cv.string,
        vol.Required(ATTR_TO_USER_ID): cv.string,
    },
    extra=vol.ALLOW_EXTRA,
)

SERVICE_REMOVE_MEASUREMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_USER_ID): cv.string,
    },
    extra=vol.ALLOW_EXTRA,
)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old config entries to new format.

    Note: We track migration state locally because entry.version is not updated
    immediately after async_update_entry() - it only reflects the stored version
    after a reload. This ensures all sequential migrations run correctly.
    """
    _LOGGER.debug(
        "Checking if migration needed for config entry version %s", entry.version
    )

    # Track current version locally (entry.version won't update after async_update_entry)
    current_version = entry.version
    new_data = {**entry.data}
    show_v2_notification = False

    # --- Migration v1 → v2: Multi-user support ---
    if current_version < 2:
        _LOGGER.info("Migrating config entry from version 1 to version 2")

        # Copy old data
        old_data = {**entry.data}

        # Safety check: If data already has v2+ structure, skip migration
        # This prevents data loss if migration is called on already-migrated entry
        if CONF_USER_PROFILES in old_data:
            _LOGGER.warning(
                "Config entry version is %s but data already has v2 structure "
                "(CONF_USER_PROFILES exists). Skipping v1→v2 migration to prevent data loss.",
                current_version,
            )
            new_data = old_data
            current_version = 2
        else:
            # Build new data structure (replaces old format entirely)
            new_data = {
                CONF_SCALE_DISPLAY_UNIT: old_data.get(
                    CONF_UNIT_SYSTEM, UnitOfMass.KILOGRAMS
                ),
                CONF_USER_PROFILES: [],
            }

            # If old config had body metrics enabled, create a default user profile
            if old_data.get(CONF_CALC_BODY_METRICS, False):
                _LOGGER.debug(
                    "Creating default user profile from legacy body metrics config"
                )

                # Use empty string as user_id to preserve entity IDs
                # This ensures sensor.etekcity_fitness_scale_ble_weight stays the same
                default_user = {
                    CONF_USER_ID: "",  # Empty string preserves entity IDs from v1
                    CONF_USER_NAME: "Default User",
                    CONF_PERSON_ENTITY: None,
                    CONF_BODY_METRICS_ENABLED: True,
                    CONF_SEX: old_data.get(CONF_SEX),
                    CONF_BIRTHDATE: old_data.get(CONF_BIRTHDATE),
                    CONF_HEIGHT: old_data.get(CONF_HEIGHT),
                    CONF_CREATED_AT: datetime.now().isoformat(),
                    CONF_UPDATED_AT: datetime.now().isoformat(),
                }

                new_data[CONF_USER_PROFILES] = [default_user]
                _LOGGER.debug("Default user profile created with body metrics enabled")
            else:
                _LOGGER.debug(
                    "Creating default user profile (no body metrics in legacy config)"
                )

                # Even without body metrics, create a default user to ensure sensors exist
                default_user = {
                    CONF_USER_ID: "",  # Empty string preserves entity IDs from v1
                    CONF_USER_NAME: "Default User",
                    CONF_PERSON_ENTITY: None,
                    CONF_BODY_METRICS_ENABLED: False,
                    CONF_CREATED_AT: datetime.now().isoformat(),
                    CONF_UPDATED_AT: datetime.now().isoformat(),
                }

                new_data[CONF_USER_PROFILES] = [default_user]

            current_version = 2
            show_v2_notification = True
            _LOGGER.info("Migration to version 2 prepared successfully")

    # --- Migration v2 → v3: Mobile notifications and weight history ---
    if current_version < 3:
        _LOGGER.info("Migrating config entry from version 2 to version 3")

        # Add mobile_notify_services and weight_history fields to each user profile
        user_profiles = new_data.get(CONF_USER_PROFILES, [])
        for user_profile in user_profiles:
            if CONF_MOBILE_NOTIFY_SERVICES not in user_profile:
                user_profile[CONF_MOBILE_NOTIFY_SERVICES] = []
                _LOGGER.debug(
                    "Added mobile_notify_services field to user %s",
                    user_profile.get(CONF_USER_NAME, "unknown"),
                )

            if CONF_WEIGHT_HISTORY not in user_profile:
                user_profile[CONF_WEIGHT_HISTORY] = []
                _LOGGER.debug(
                    "Added weight_history field to user %s",
                    user_profile.get(CONF_USER_NAME, "unknown"),
                )

        current_version = 3
        _LOGGER.info("Migration to version 3 prepared successfully")

    # Save all migrations at once with final version
    if entry.version != current_version:
        hass.config_entries.async_update_entry(
            entry, data=new_data, version=current_version
        )
        _LOGGER.info(
            "Config entry migration completed: v%s → v%s",
            entry.version,
            current_version,
        )

    # Show notification after save (only for v1→v2 upgrade)
    if show_v2_notification:
        from homeassistant.components import persistent_notification

        persistent_notification.create(
            hass,
            "Your Etekcity Fitness Scale has been upgraded to support multiple users!\n\n"
            "Your existing sensors and historical data have been preserved under a default user profile.\n\n"
            "You can now:\n"
            "- Rename the default user\n"
            "- Link it to a person entity\n"
            "- Add additional users\n"
            "- Remove the default user (once you've added other users)\n\n"
            "Go to the integration's **Device Settings** (⚙️) to manage user profiles.",
            title="Etekcity Scale: Multi-User Support Enabled",
            notification_id="etekcity_scale_migration_v2",
        )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up scale from a config entry.

    Note: async_migrate_entry is called automatically by Home Assistant before this
    function when the entry version is less than the current version. We don't need
    to call it again here.
    """
    hass.data.setdefault(DOMAIN, {})
    address = entry.unique_id

    assert address is not None
    await close_stale_connections_by_address(address)

    # Get user profiles from config entry
    user_profiles = deepcopy(entry.data.get(CONF_USER_PROFILES, []))

    coordinator = ScaleDataUpdateCoordinator(hass, address, user_profiles, entry.title)
    coordinator.set_config_entry_id(entry.entry_id)

    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Helper: resolve device_id -> coordinator and optionally validate user IDs
    def _get_coordinator_for_device(device_id: str) -> ScaleDataUpdateCoordinator:
        """Return coordinator for a specific device_id or raise HomeAssistantError."""
        from homeassistant.helpers import device_registry as dr
        from homeassistant.exceptions import HomeAssistantError

        device_registry = dr.async_get(hass)
        device_entry = device_registry.async_get(device_id)
        if not device_entry:
            raise HomeAssistantError(f"Device {device_id} not found")

        for entry_id in device_entry.config_entries:
            coord = hass.data.get(DOMAIN, {}).get(entry_id)
            if isinstance(coord, ScaleDataUpdateCoordinator):
                return coord

        raise HomeAssistantError(
            f"Device {device_entry.name or device_id} is not an Etekcity Fitness Scale"
        )

    def _get_single_device_id(call: ServiceCall) -> str:
        """Extract a single device_id from a service call."""
        from homeassistant.exceptions import HomeAssistantError

        device_id = call.data.get("device_id")

        if not device_id:
            raise HomeAssistantError(
                "No scale device specified. Include a `device_id` field in the service call."
            )
        if not isinstance(device_id, str):
            raise HomeAssistantError("`device_id` must be a string.")

        return device_id

    def _resolve_user_id(user_id_input: str | None) -> str:
        """Resolve user_id input, mapping quotes or whitespace to empty string.

        Args:
            user_id_input: The user_id from service call

        Returns:
            The actual user_id (empty string for legacy user, or the input value)
        """
        if not user_id_input:
            return ""

        val = user_id_input.strip("\"' ")

        return val

    def _format_user_list(profiles: list[dict]) -> str:
        """Format user profiles for display in error messages."""
        user_list_items = []
        for p in profiles:
            profile_user_id = p.get(CONF_USER_ID)
            if profile_user_id is not None:
                user_name = p.get(CONF_USER_NAME, "Unknown")
                if profile_user_id == "":
                    user_list_items.append(f'{user_name} (leave blank or use "")')
                else:
                    user_list_items.append(f"{user_name} (user_id: {profile_user_id})")
        return ", ".join(user_list_items) if user_list_items else "none"

    # Register services
    async def handle_assign_measurement(call: ServiceCall) -> None:
        """Handle the assign_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        device_id = _get_single_device_id(call)

        timestamp = call.data[ATTR_TIMESTAMP]
        # Resolve user_id: "default" keyword maps to empty string (legacy user)
        user_id = _resolve_user_id(call.data.get(ATTR_USER_ID))

        _LOGGER.debug(
            "Service call assign_measurement on device %s timestamp=%s user_id=%s",
            device_id,
            timestamp,
            repr(user_id),  # Use repr to show empty string clearly in logs
        )

        coord = _get_coordinator_for_device(device_id)

        # Validate user exists on this scale
        # Note: Must check 'is not None' to include empty string (legacy v1 user_id)
        valid_user_ids = [
            p.get(CONF_USER_ID)
            for p in coord.get_user_profiles()
            if p.get(CONF_USER_ID) is not None
        ]
        if user_id not in valid_user_ids:
            available_users = _format_user_list(coord.get_user_profiles())
            if user_id == "":
                raise HomeAssistantError(
                    f"No default user found for this scale. "
                    f"Please specify a user_id. "
                    f"Available users: {available_users}"
                )
            else:
                raise HomeAssistantError(
                    f"User '{user_id}' not found for this scale. "
                    f"Please check the user ID and try again. "
                    f"Available users: {available_users}"
                )

        # Validate timestamp exists in pending measurements
        pending_measurements = coord.get_pending_measurements()
        if timestamp not in pending_measurements:
            available_timestamps = sorted(pending_measurements.keys(), reverse=True)[:5]
            if available_timestamps:
                timestamp_list = ", ".join(f"'{ts}'" for ts in available_timestamps)
                raise HomeAssistantError(
                    f"Measurement timestamp '{timestamp}' not found. "
                    f"Please check the timestamp and try again. "
                    f"Available timestamps: {timestamp_list}"
                )
            else:
                raise HomeAssistantError(
                    f"Measurement timestamp '{timestamp}' not found. "
                    f"No pending measurements are available for this scale."
                )

        # Assign the pending measurement
        if not coord.assign_pending_measurement(timestamp, user_id):
            raise HomeAssistantError(
                f"Failed to assign measurement to user '{user_id}'. "
                f"Please check that the timestamp '{timestamp}' exists in pending measurements."
            )

    async def handle_reassign_measurement(call: ServiceCall) -> None:
        """Handle the reassign_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        device_id = _get_single_device_id(call)

        # Resolve user_ids: "default" keyword maps to empty string (legacy user)
        from_user_id = _resolve_user_id(call.data.get(ATTR_FROM_USER_ID))
        to_user_id = _resolve_user_id(call.data.get(ATTR_TO_USER_ID))
        timestamp = call.data.get(ATTR_TIMESTAMP)  # Optional

        coord = _get_coordinator_for_device(device_id)

        # Validate both users exist on this scale
        # Note: Must check 'is not None' to include empty string (legacy v1 user_id)
        valid_user_ids = [
            p.get(CONF_USER_ID)
            for p in coord.get_user_profiles()
            if p.get(CONF_USER_ID) is not None
        ]
        for uid, field_name in [
            (from_user_id, "from_user_id"),
            (to_user_id, "to_user_id"),
        ]:
            if uid not in valid_user_ids:
                available_users = _format_user_list(coord.get_user_profiles())
                if uid == "":
                    raise HomeAssistantError(
                        f"No default user found for this scale ({field_name}). "
                        f"Please specify a user_id. "
                        f"Available users: {available_users}"
                    )
                else:
                    raise HomeAssistantError(
                        f"User '{uid}' not found for this scale ({field_name}). "
                        f"Please check the user ID and try again. "
                        f"Available users: {available_users}"
                    )

        if not coord.reassign_user_measurement(from_user_id, to_user_id, timestamp):
            if timestamp:
                raise HomeAssistantError(
                    f"Failed to reassign measurement. "
                    f"Please ensure user '{from_user_id}' has a measurement with timestamp '{timestamp}'."
                )
            else:
                raise HomeAssistantError(
                    f"Failed to reassign measurement. "
                    f"Please ensure user '{from_user_id}' has at least one measurement to reassign."
                )

    async def handle_remove_measurement(call: ServiceCall) -> None:
        """Handle the remove_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        device_id = _get_single_device_id(call)

        user_id = _resolve_user_id(call.data.get(ATTR_USER_ID))
        timestamp = call.data.get(ATTR_TIMESTAMP)  # Optional
        coord = _get_coordinator_for_device(device_id)

        # Validate user exists on this scale
        # Note: Must check 'is not None' to include empty string (legacy v1 user_id)
        valid_user_ids = [
            p.get(CONF_USER_ID)
            for p in coord.get_user_profiles()
            if p.get(CONF_USER_ID) is not None
        ]
        if user_id not in valid_user_ids:
            available_users = _format_user_list(coord.get_user_profiles())
            if user_id == "":
                raise HomeAssistantError(
                    f"No default user found for this scale. "
                    f"Please specify a user_id. "
                    f"Available users: {available_users}"
                )
            else:
                raise HomeAssistantError(
                    f"User '{user_id}' not found for this scale. "
                    f"Please check the user ID and try again. "
                    f"Available users: {available_users}"
                )

        if not coord.remove_user_measurement(user_id, timestamp):
            if timestamp:
                raise HomeAssistantError(
                    f"Failed to remove measurement. "
                    f"Please ensure user '{user_id}' has a measurement with timestamp '{timestamp}'."
                )
            else:
                raise HomeAssistantError(
                    f"Failed to remove measurement. "
                    f"Please ensure user '{user_id}' has at least one measurement to remove."
                )

    # Register the services on first setup
    if not hass.services.has_service(DOMAIN, SERVICE_ASSIGN_MEASUREMENT):
        hass.services.async_register(
            DOMAIN,
            SERVICE_ASSIGN_MEASUREMENT,
            handle_assign_measurement,
            schema=SERVICE_ASSIGN_MEASUREMENT_SCHEMA,
        )
        _LOGGER.debug("Registered service: %s.%s", DOMAIN, SERVICE_ASSIGN_MEASUREMENT)

    if not hass.services.has_service(DOMAIN, SERVICE_REASSIGN_MEASUREMENT):
        hass.services.async_register(
            DOMAIN,
            SERVICE_REASSIGN_MEASUREMENT,
            handle_reassign_measurement,
            schema=SERVICE_REASSIGN_MEASUREMENT_SCHEMA,
        )
        _LOGGER.debug("Registered service: %s.%s", DOMAIN, SERVICE_REASSIGN_MEASUREMENT)

    if not hass.services.has_service(DOMAIN, SERVICE_REMOVE_MEASUREMENT):
        hass.services.async_register(
            DOMAIN,
            SERVICE_REMOVE_MEASUREMENT,
            handle_remove_measurement,
            schema=SERVICE_REMOVE_MEASUREMENT_SCHEMA,
        )
        _LOGGER.debug("Registered service: %s.%s", DOMAIN, SERVICE_REMOVE_MEASUREMENT)

    # Register event listener for mobile app notification actions
    async def handle_mobile_app_notification_action(event):
        """Handle mobile app notification action events."""
        action = event.data.get("action")

        if not action or not isinstance(action, str):
            return

        # Only handle our scale notification actions
        if not action.startswith("SCALE_"):
            return

        def _decode_action_payload(
            action_value: str, prefix: str
        ) -> tuple[str, str] | None:
            """Extract user_id and timestamp tokens from an action string.

            Handles v1 legacy user_id (empty string) by using __legacy__ placeholder.
            """
            if not action_value.startswith(prefix):
                return None

            payload = action_value[len(prefix) :]
            if "_" not in payload:
                return None

            user_token, timestamp_token = payload.rsplit("_", 1)
            if not user_token or not timestamp_token:
                return None

            # Decode user_id, handling v1 legacy placeholder
            LEGACY_USER_ID_PLACEHOLDER = "__legacy__"
            decoded_user_id = unquote(user_token)
            if decoded_user_id == LEGACY_USER_ID_PLACEHOLDER:
                decoded_user_id = ""  # Map placeholder back to empty string (v1 legacy)

            return decoded_user_id, unquote(timestamp_token)

        try:
            if action.startswith("SCALE_ASSIGN_"):
                decoded = _decode_action_payload(action, "SCALE_ASSIGN_")
                if not decoded:
                    _LOGGER.warning("Invalid SCALE_ASSIGN action format: %s", action)
                    return

                user_id, timestamp = decoded

                _LOGGER.info(
                    "Mobile app action: assigning measurement %s to user %s",
                    timestamp,
                    user_id,
                )

                # Find the coordinator that has this pending measurement
                assigned = False
                for coord in hass.data.get(DOMAIN, {}).values():
                    if not isinstance(coord, ScaleDataUpdateCoordinator):
                        continue

                    if timestamp in coord.get_pending_measurements():
                        # Validate user exists in this coordinator
                        # Note: Must check 'is not None' to include empty string (legacy v1 user_id)
                        valid_user_ids = [
                            p.get(CONF_USER_ID)
                            for p in coord.get_user_profiles()
                            if p.get(CONF_USER_ID) is not None
                        ]

                        if user_id not in valid_user_ids:
                            _LOGGER.error(
                                "User %s not found in coordinator for timestamp %s",
                                user_id,
                                timestamp,
                            )
                            continue

                        # Assign the measurement
                        if coord.assign_pending_measurement(timestamp, user_id):
                            assigned = True
                            _LOGGER.info(
                                "Successfully assigned measurement %s to user %s via mobile app",
                                timestamp,
                                user_id,
                            )
                        break

                if not assigned:
                    _LOGGER.debug(
                        "Could not assign measurement %s to user %s - "
                        "measurement may have been already assigned or expired",
                        timestamp,
                        user_id,
                    )

            elif action.startswith("SCALE_NOT_ME_"):
                decoded = _decode_action_payload(action, "SCALE_NOT_ME_")
                if not decoded:
                    _LOGGER.warning("Invalid SCALE_NOT_ME action format: %s", action)
                    return

                user_id, timestamp = decoded

                # For "Not Me" action, we just log it
                # The notification is already dismissed by the mobile app
                _LOGGER.debug(
                    "User %s indicated measurement %s is not theirs (via mobile app)",
                    user_id,
                    timestamp,
                )
                # No further action needed - measurement remains pending for other users
            else:
                _LOGGER.warning("Unknown scale action: %s", action)

        except Exception as ex:
            _LOGGER.error("Error handling mobile app notification action: %s", ex)

    # Register the event listener once (shared across all entries)
    if DATA_MOBILE_APP_LISTENER_UNSUB not in hass.data[DOMAIN]:
        hass.data[DOMAIN][DATA_MOBILE_APP_LISTENER_UNSUB] = hass.bus.async_listen(
            "mobile_app_notification_action", handle_mobile_app_notification_action
        )
        _LOGGER.debug("Registered event listener for mobile_app_notification_action")

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(coordinator.async_stop)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        coordinator: ScaleDataUpdateCoordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_stop()
        bluetooth.async_rediscover_address(hass, coordinator.address)

        remaining_coordinators = [
            value
            for value in hass.data[DOMAIN].values()
            if isinstance(value, ScaleDataUpdateCoordinator)
        ]

        # Unregister shared listener and services if this is the last entry
        if not remaining_coordinators:
            if unsub := hass.data[DOMAIN].pop(DATA_MOBILE_APP_LISTENER_UNSUB, None):
                unsub()
                _LOGGER.debug(
                    "Unregistered event listener for mobile_app_notification_action"
                )
            hass.services.async_remove(DOMAIN, SERVICE_ASSIGN_MEASUREMENT)
            _LOGGER.debug(
                "Unregistered service: %s.%s", DOMAIN, SERVICE_ASSIGN_MEASUREMENT
            )
            hass.services.async_remove(DOMAIN, SERVICE_REASSIGN_MEASUREMENT)
            _LOGGER.debug(
                "Unregistered service: %s.%s", DOMAIN, SERVICE_REASSIGN_MEASUREMENT
            )
            hass.services.async_remove(DOMAIN, SERVICE_REMOVE_MEASUREMENT)
            _LOGGER.debug(
                "Unregistered service: %s.%s", DOMAIN, SERVICE_REMOVE_MEASUREMENT
            )

    return unload_ok
