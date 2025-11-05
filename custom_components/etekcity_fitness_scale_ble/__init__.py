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

# Service schemas (target is defined in services.yaml, not in vol.Schema)
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
        """Extract device_id from service call target."""
        from homeassistant.exceptions import HomeAssistantError

        device_ids = []

        # When target: is defined in services.yaml, Home Assistant processes it
        # and makes it available via call.service_data or call.data
        # Try multiple locations where it might be stored
        
        # 1. Check call.service_data (processed target data)
        if hasattr(call, "service_data"):
            service_data = call.service_data
            if isinstance(service_data, dict):
                target = service_data.get("target", {})
                if isinstance(target, dict):
                    device_ids = target.get("device_id", [])
        
        # 2. Check call.data["target"] (raw YAML format)
        if not device_ids:
            target = call.data.get("target")
            if isinstance(target, dict):
                # Handle both formats:
                # target: {device_id: "abc123"} or target: {device_id: ["abc123"]}
                device_id_val = target.get("device_id")
                if device_id_val:
                    if isinstance(device_id_val, list):
                        device_ids = device_id_val
                    elif isinstance(device_id_val, str):
                        device_ids = [device_id_val]
            elif isinstance(target, list):
                device_ids = target
        
        # 3. Check call.target attribute (if ServiceCall has it)
        if not device_ids and hasattr(call, "target"):
            target_obj = call.target
            if hasattr(target_obj, "device_id"):
                device_id_val = target_obj.device_id
                device_ids = device_id_val if isinstance(device_id_val, list) else [device_id_val]
            elif hasattr(target_obj, "device_ids"):
                device_ids = target_obj.device_ids
        
        # 4. Check if device_id is directly in call.data (fallback for manual calls)
        if not device_ids:
            device_id = call.data.get("device_id")
            if device_id:
                device_ids = [device_id] if isinstance(device_id, str) else device_id

        # Ensure it's a list
        if not isinstance(device_ids, list):
            device_ids = [device_ids] if device_ids else []

        if not device_ids:
            # Debug: log what we actually received to help diagnose
            _LOGGER.debug(
                "Service call - data: %s, has service_data: %s, has target attr: %s",
                call.data,
                hasattr(call, "service_data"),
                hasattr(call, "target"),
            )
            if hasattr(call, "service_data"):
                _LOGGER.debug("Service call service_data: %s", call.service_data)
            raise HomeAssistantError(
                "No scale device specified. Use 'target' with a device_id in the service call. "
                "Example YAML format:\n"
                "target:\n"
                "  device_id: ['your-device-id']\n"
                "data:\n"
                "  timestamp: '...'\n"
                "  user_id: '...'"
            )
        if len(device_ids) != 1:
            raise HomeAssistantError(
                f"Exactly one scale device must be specified in target, got {len(device_ids)}"
            )
        return str(device_ids[0])

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
        valid_user_ids = [p.get(CONF_USER_ID) for p in coord.get_user_profiles()]
        if user_id not in valid_user_ids:
            raise HomeAssistantError(f"User {user_id} not found for selected scale")

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

        for uid in (from_user_id, to_user_id):
            if uid not in [p.get(CONF_USER_ID) for p in coord.get_user_profiles()]:
                raise HomeAssistantError(f"User {uid} not found for selected scale")

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

        if user_id not in [p.get(CONF_USER_ID) for p in coord.get_user_profiles()]:
            raise HomeAssistantError(f"User {user_id} not found for selected scale")

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
