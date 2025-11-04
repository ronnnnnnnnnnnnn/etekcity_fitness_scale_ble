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
    }
)

SERVICE_REASSIGN_MEASUREMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_FROM_USER_ID): cv.string,
        vol.Required(ATTR_TO_USER_ID): cv.string,
    }
)

SERVICE_REMOVE_MEASUREMENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_USER_ID): cv.string,
    }
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

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up scale from a config entry."""

    # Migrate config entry if needed
    if entry.version < 2:
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

    # Register services
    async def handle_assign_measurement(call: ServiceCall) -> None:
        """Handle the assign_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        timestamp = call.data[ATTR_TIMESTAMP]
        user_id = call.data[ATTR_USER_ID]

        _LOGGER.debug(
            "Service call: assign_measurement(timestamp=%s, user_id=%s)",
            timestamp,
            user_id,
        )

        # Find the coordinator for this scale
        # For now, use the first coordinator (single scale assumption)
        # In multi-scale setups, would need device_id in service call
        coordinator_found = False
        for coord in hass.data[DOMAIN].values():
            if isinstance(coord, ScaleDataUpdateCoordinator):
                # Validate user exists
                user_profiles = coord.get_user_profiles()
                user_ids = [
                    profile.get(CONF_USER_ID)
                    for profile in user_profiles
                    if profile.get(CONF_USER_ID)
                ]
                if user_id not in user_ids:
                    raise HomeAssistantError(
                        f"User {user_id} not found in user profiles"
                    )

                success = coord.assign_pending_measurement(timestamp, user_id)
                if success:
                    _LOGGER.info(
                        "Successfully assigned measurement %s to user %s",
                        timestamp,
                        user_id,
                    )
                else:
                    _LOGGER.warning(
                        "Failed to assign measurement %s to user %s",
                        timestamp,
                        user_id,
                    )
                    raise HomeAssistantError(
                        f"Failed to assign measurement {timestamp} to user {user_id}. "
                        f"Check that the timestamp exists in pending measurements."
                    )
                coordinator_found = True
                break

        if not coordinator_found:
            _LOGGER.error("No coordinator found for assign_measurement service")
            raise HomeAssistantError(
                "No scale found. Ensure the Etekcity Fitness Scale integration is set up."
            )

    async def handle_reassign_measurement(call: ServiceCall) -> None:
        """Handle the reassign_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        from_user_id = call.data[ATTR_FROM_USER_ID]
        to_user_id = call.data[ATTR_TO_USER_ID]

        _LOGGER.debug(
            "Service call: reassign_measurement(from_user_id=%s, to_user_id=%s)",
            from_user_id,
            to_user_id,
        )

        # Find the coordinator for this scale
        coordinator_found = False
        for coord in hass.data[DOMAIN].values():
            if isinstance(coord, ScaleDataUpdateCoordinator):
                # Validate users exist
                user_profiles = coord.get_user_profiles()
                user_ids = [
                    profile.get(CONF_USER_ID)
                    for profile in user_profiles
                    if profile.get(CONF_USER_ID)
                ]
                if from_user_id not in user_ids:
                    raise HomeAssistantError(
                        f"Source user {from_user_id} not found in user profiles"
                    )
                if to_user_id not in user_ids:
                    raise HomeAssistantError(
                        f"Target user {to_user_id} not found in user profiles"
                    )

                success = coord.reassign_user_measurement(from_user_id, to_user_id)
                if success:
                    _LOGGER.info(
                        "Successfully reassigned measurement from user %s to user %s",
                        from_user_id,
                        to_user_id,
                    )
                else:
                    _LOGGER.warning(
                        "Failed to reassign measurement from user %s to user %s",
                        from_user_id,
                        to_user_id,
                    )
                    raise HomeAssistantError(
                        f"Failed to reassign measurement from user {from_user_id} to user {to_user_id}. "
                        f"Check that the source user has a recent measurement."
                    )
                coordinator_found = True
                break

        if not coordinator_found:
            _LOGGER.error("No coordinator found for reassign_measurement service")
            raise HomeAssistantError(
                "No scale found. Ensure the Etekcity Fitness Scale integration is set up."
            )

    async def handle_remove_measurement(call: ServiceCall) -> None:
        """Handle the remove_measurement service call."""
        from homeassistant.exceptions import HomeAssistantError

        user_id = call.data[ATTR_USER_ID]

        _LOGGER.debug(
            "Service call: remove_measurement(user_id=%s)",
            user_id,
        )

        # Find the coordinator for this scale
        coordinator_found = False
        for coord in hass.data[DOMAIN].values():
            if isinstance(coord, ScaleDataUpdateCoordinator):
                # Validate user exists
                user_profiles = coord.get_user_profiles()
                user_ids = [
                    profile.get(CONF_USER_ID)
                    for profile in user_profiles
                    if profile.get(CONF_USER_ID)
                ]
                if user_id not in user_ids:
                    raise HomeAssistantError(
                        f"User {user_id} not found in user profiles"
                    )

                success = coord.remove_user_measurement(user_id)
                if success:
                    _LOGGER.info(
                        "Successfully removed measurement for user %s",
                        user_id,
                    )
                else:
                    _LOGGER.warning(
                        "Failed to remove measurement for user %s",
                        user_id,
                    )
                    raise HomeAssistantError(
                        f"Failed to remove measurement for user {user_id}. "
                        f"Check that the user has a recent measurement to remove."
                    )
                coordinator_found = True
                break

        if not coordinator_found:
            _LOGGER.error("No coordinator found for remove_measurement service")
            raise HomeAssistantError(
                "No scale found. Ensure the Etekcity Fitness Scale integration is set up."
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
