"""The etekcity_fitness_scale_ble integration."""

from __future__ import annotations

from datetime import datetime
import logging

import voluptuous as vol

from bleak_retry_connector import close_stale_connections_by_address
from homeassistant.components import bluetooth
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_UNIT_SYSTEM, Platform, UnitOfMass
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv

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
    DOMAIN,
)
from .coordinator import ScaleDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR]

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
    """Migrate old config entries to new format."""

    _LOGGER.debug(
        "Checking if migration needed for config entry version %s", entry.version
    )

    if entry.version == 1:
        _LOGGER.info("Migrating config entry from version 1 to version 2")

        # Copy old data
        old_data = {**entry.data}

        # Build new data structure
        new_data = {
            CONF_SCALE_DISPLAY_UNIT: old_data.pop(
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

        # Update the entry
        hass.config_entries.async_update_entry(entry, data=new_data, version=2)
        _LOGGER.info("Migration of config entry to version 2 completed successfully")

        # Create a persistent notification to inform the user about the upgrade
        from homeassistant.components import persistent_notification

        persistent_notification.create(
            hass,
            "Your Etekcity Fitness Scale has been upgraded to support multiple users!\n\n"
            "Your existing sensors and historical data have been preserved under a default user profile.\n\n"
            "You can now:\n"
            "- Rename the default user\n"
            "- Link it to a person entity\n"
            "- Add additional users\n\n"
            "Go to **Settings → Devices & Services → Etekcity Fitness Scale BLE → Configure** to manage user profiles.",
            title="Etekcity Scale: Multi-User Support Enabled",
            notification_id="etekcity_scale_migration_v2",
        )

    if entry.version == 2:
        _LOGGER.info("Migrating config entry from version 2 to version 3")

        # Copy existing data
        new_data = {**entry.data}

        # Add mobile_notify_services field to each user profile
        user_profiles = new_data.get(CONF_USER_PROFILES, [])
        for user_profile in user_profiles:
            if CONF_MOBILE_NOTIFY_SERVICES not in user_profile:
                user_profile[CONF_MOBILE_NOTIFY_SERVICES] = []
                _LOGGER.debug(
                    "Added mobile_notify_services field to user %s",
                    user_profile.get(CONF_USER_NAME, "unknown"),
                )

        # Update the entry
        hass.config_entries.async_update_entry(entry, data=new_data, version=3)
        _LOGGER.info("Migration of config entry to version 3 completed successfully")

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up scale from a config entry."""

    # Migrate config entry if needed
    if entry.version < 3:
        if not await async_migrate_entry(hass, entry):
            _LOGGER.error("Migration failed for config entry")
            return False

    hass.data.setdefault(DOMAIN, {})
    address = entry.unique_id

    assert address is not None
    await close_stale_connections_by_address(address)

    # Get user profiles from config entry
    user_profiles = entry.data.get(CONF_USER_PROFILES, [])

    coordinator = ScaleDataUpdateCoordinator(hass, address, user_profiles, entry.title)

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

    # Register services
    async def handle_assign_measurement(call: ServiceCall) -> None:
        """Handle the assign_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        device_id = _get_single_device_id(call)

        timestamp = call.data[ATTR_TIMESTAMP]
        user_id = call.data[ATTR_USER_ID]

        _LOGGER.debug(
            "Service call assign_measurement on device %s timestamp=%s user_id=%s",
            device_id,
            timestamp,
            user_id,
        )

        coord = _get_coordinator_for_device(device_id)

        # Validate user exists on this scale
        valid_user_ids = [
            p.get(CONF_USER_ID)
            for p in coord.get_user_profiles()
            if p.get(CONF_USER_ID)
        ]
        if user_id not in valid_user_ids:
            # Provide helpful error with available users
            user_names = [
                f"{p.get(CONF_USER_NAME, 'Unknown')} ({p.get(CONF_USER_ID)})"
                for p in coord.get_user_profiles()
                if p.get(CONF_USER_ID)
            ]
            raise HomeAssistantError(
                f"User {user_id} not found for selected scale. "
                f"Available users: {', '.join(user_names) if user_names else 'none'}"
            )

        # Validate timestamp exists in pending measurements
        pending_measurements = coord.get_pending_measurements()
        if timestamp not in pending_measurements:
            available_timestamps = sorted(pending_measurements.keys(), reverse=True)[:5]
            timestamp_list = ", ".join(f"'{ts}'" for ts in available_timestamps)
            raise HomeAssistantError(
                f"Timestamp {timestamp} not found in pending measurements for this scale. "
                f"{f'Available timestamps: {timestamp_list}' if available_timestamps else 'No pending measurements available.'}"
            )

        # Assign the pending measurement
        if not coord.assign_pending_measurement(timestamp, user_id):
            raise HomeAssistantError(
                f"Failed assigning timestamp {timestamp} to user {user_id}. "
                "Check pending measurements for this scale."
            )

    async def handle_reassign_measurement(call: ServiceCall) -> None:
        """Handle the reassign_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        device_id = _get_single_device_id(call)

        from_user_id = call.data[ATTR_FROM_USER_ID]
        to_user_id = call.data[ATTR_TO_USER_ID]

        coord = _get_coordinator_for_device(device_id)

        # Validate both users exist on this scale
        valid_user_ids = [
            p.get(CONF_USER_ID)
            for p in coord.get_user_profiles()
            if p.get(CONF_USER_ID)
        ]
        for uid in (from_user_id, to_user_id):
            if uid not in valid_user_ids:
                user_names = [
                    f"{p.get(CONF_USER_NAME, 'Unknown')} ({p.get(CONF_USER_ID)})"
                    for p in coord.get_user_profiles()
                    if p.get(CONF_USER_ID)
                ]
                raise HomeAssistantError(
                    f"User {uid} not found for selected scale. "
                    f"Available users: {', '.join(user_names) if user_names else 'none'}"
                )

        if not coord.reassign_user_measurement(from_user_id, to_user_id):
            raise HomeAssistantError(
                "Failed to reassign measurement. Ensure source user has a recent measurement."
            )

    async def handle_remove_measurement(call: ServiceCall) -> None:
        """Handle the remove_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        device_id = _get_single_device_id(call)

        user_id = call.data[ATTR_USER_ID]
        coord = _get_coordinator_for_device(device_id)

        # Validate user exists on this scale
        valid_user_ids = [
            p.get(CONF_USER_ID)
            for p in coord.get_user_profiles()
            if p.get(CONF_USER_ID)
        ]
        if user_id not in valid_user_ids:
            user_names = [
                f"{p.get(CONF_USER_NAME, 'Unknown')} ({p.get(CONF_USER_ID)})"
                for p in coord.get_user_profiles()
                if p.get(CONF_USER_ID)
            ]
            raise HomeAssistantError(
                f"User {user_id} not found for selected scale. "
                f"Available users: {', '.join(user_names) if user_names else 'none'}"
            )

        if not coord.remove_user_measurement(user_id):
            raise HomeAssistantError(
                "Failed to remove measurement. Ensure the user has a recent measurement."
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

        try:
            # Parse action: SCALE_ASSIGN_{user_id}_{timestamp} or SCALE_NOT_ME_{user_id}_{timestamp}
            parts = action.split(
                "_", 3
            )  # Split into ["SCALE", "ASSIGN"/"NOT", "ME"/{user_id}, {timestamp}]

            if len(parts) < 3:
                _LOGGER.warning("Invalid scale action format: %s", action)
                return

            if parts[1] == "ASSIGN":
                # SCALE_ASSIGN_{user_id}_{timestamp}
                if len(parts) != 4:
                    _LOGGER.warning("Invalid SCALE_ASSIGN action format: %s", action)
                    return

                user_id = parts[2]
                timestamp = parts[3]

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
                        valid_user_ids = [
                            p.get(CONF_USER_ID)
                            for p in coord.get_user_profiles()
                            if p.get(CONF_USER_ID)
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
                    _LOGGER.warning(
                        "Could not assign measurement %s to user %s - "
                        "measurement may have been already assigned or expired",
                        timestamp,
                        user_id,
                    )

            elif parts[1] == "NOT" and parts[2] == "ME":
                # SCALE_NOT_ME_{user_id}_{timestamp}
                if len(parts) != 4:
                    _LOGGER.warning("Invalid SCALE_NOT_ME action format: %s", action)
                    return

                user_id = parts[3].rsplit("_", 1)[0] if "_" in parts[3] else parts[3]
                timestamp = parts[3].rsplit("_", 1)[-1] if "_" in parts[3] else ""

                # For "Not Me" action, we just log it
                # The notification is already dismissed by the mobile app
                _LOGGER.debug(
                    "User %s indicated measurement %s is not theirs (via mobile app)",
                    user_id,
                    timestamp,
                )
                # No further action needed - measurement remains pending for other users

        except Exception as ex:
            _LOGGER.error("Error handling mobile app notification action: %s", ex)

    # Register the event listener
    hass.bus.async_listen(
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

        # Unregister services if this is the last entry
        if not hass.data[DOMAIN]:
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
