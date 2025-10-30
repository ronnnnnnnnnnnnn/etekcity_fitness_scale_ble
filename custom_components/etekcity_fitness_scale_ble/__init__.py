"""The etekcity_fitness_scale_ble integration."""

from __future__ import annotations

from datetime import datetime
import logging
import uuid

from bleak_retry_connector import close_stale_connections_by_address
from homeassistant.components import bluetooth
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_UNIT_SYSTEM, Platform, UnitOfMass
from homeassistant.core import HomeAssistant

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

            default_user = {
                CONF_USER_ID: str(uuid.uuid4()),
                CONF_USER_NAME: "User",
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
                "No body metrics in legacy config, starting with empty user profiles"
            )

        # Update the entry
        hass.config_entries.async_update_entry(entry, data=new_data, version=2)
        _LOGGER.info("Migration to version 2 completed successfully")

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

    coordinator = ScaleDataUpdateCoordinator(hass, address)

    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(coordinator.async_stop)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        coordinator: ScaleDataUpdateCoordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_stop()
        bluetooth.async_rediscover_address(hass, coordinator.address)

    return unload_ok
