from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Self

from etekcity_esf551_ble import WEIGHT_KEY, WeightUnit
from homeassistant import config_entries
from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntityDescription,
    SensorExtraStoredData,
    SensorStateClass,
)
from homeassistant.const import UnitOfMass
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import (
    CONNECTION_BLUETOOTH,
    DeviceInfo,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import UNDEFINED
from sensor_state_data import DeviceClass, Units

from .const import DOMAIN
from .coordinator import ScaleData, ScaleDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

SENSOR_DESCRIPTIONS = {
    # Impedance sensor (ohm)
    (DeviceClass.IMPEDANCE, Units.OHM): SensorEntityDescription(
        key=f"{DeviceClass.IMPEDANCE}_{Units.OHM}",
        icon="mdi:omega",
        native_unit_of_measurement=Units.OHM,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    # Mass sensor (kg)
    (DeviceClass.MASS, Units.MASS_KILOGRAMS): SensorEntityDescription(
        key=f"{DeviceClass.MASS}_{Units.MASS_KILOGRAMS}",
        device_class=SensorDeviceClass.WEIGHT,
        native_unit_of_measurement=UnitOfMass.KILOGRAMS,
        state_class=SensorStateClass.MEASUREMENT,
    ),
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: config_entries.ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the scale sensors."""
    _LOGGER.debug("Setting up scale sensors for entry: %s", entry.entry_id)
    address = entry.unique_id
    sensors_mapping = SENSOR_DESCRIPTIONS.copy()
    coordinator = hass.data[DOMAIN][entry.entry_id]

    entity = ScaleSensor(
        entry.title,
        address,
        coordinator,
        sensors_mapping[(DeviceClass.MASS, Units.MASS_KILOGRAMS)],
    )

    async_add_entities([entity])
    _LOGGER.debug(
        "Scale sensors setup completed for entry: %s", entry.entry_id
    )


HW_VERSION_KEY = "hw_version"
SW_VERSION_KEY = "sw_version"
DISPLAY_UNIT_KEY = "display_unit"


@dataclass
class ScaleSensorExtraStoredData(SensorExtraStoredData):
    """Object to hold extra stored data for the scale sensor."""

    display_unit: str
    hw_version: str
    sw_version: str

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation of the scale sensor data."""
        data = super().as_dict()
        data[DISPLAY_UNIT_KEY] = self.display_unit
        data[HW_VERSION_KEY] = self.hw_version
        data[SW_VERSION_KEY] = self.sw_version

        return data

    @classmethod
    def from_dict(cls, restored: dict[str, Any]) -> Self | None:
        """Initialize a stored sensor state from a dict."""
        extra = SensorExtraStoredData.from_dict(restored)
        if extra is None:
            return None

        display_unit: str = restored.get(DISPLAY_UNIT_KEY)
        if not display_unit:
            display_unit = UNDEFINED

        restored.setdefault("")
        hw_version: str = restored.get(HW_VERSION_KEY)
        sw_version: str = restored.get(SW_VERSION_KEY)

        return cls(
            extra.native_value,
            extra.native_unit_of_measurement,
            display_unit,
            hw_version,
            sw_version,
        )


class ScaleSensor(RestoreSensor):
    """Representation of a sensor for the Etekcity scale."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_available = False
    _attr_device_class = SensorDeviceClass.WEIGHT
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfMass.KILOGRAMS

    def __init__(
        self,
        name: str,
        address: str,
        coordinator: ScaleDataUpdateCoordinator,
        entity_description: SensorEntityDescription,
    ) -> None:
        """Initialize the scale sensor.

        Args:
            name: The name of the sensor.
            address: The Bluetooth address of the scale.
            coordinator: The data update coordinator for the scale.
            entity_description: Description of the sensor entity.

        """
        self.entity_description = entity_description

        title = f"{name} {address}"

        self._attr_unique_id = f"{title}_{entity_description.key}"

        self._id = address
        self._attr_device_info = DeviceInfo(
            connections={(CONNECTION_BLUETOOTH, address)},
            name=name,
            manufacturer="Etekcity",
        )
        self._coordinator = coordinator

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""

        _LOGGER.debug("Adding sensor to Home Assistant: %s", self.entity_id)
        await super().async_added_to_hass()

        if last_state := await self.async_get_last_sensor_data():
            _LOGGER.debug(
                "Restoring previous state for sensor: %s", self.entity_id
            )
            self._attr_native_value = last_state.native_value
            self._attr_native_unit_of_measurement = (
                last_state.native_unit_of_measurement
            )
            self._sensor_option_unit_of_measurement = last_state.display_unit

            address = self._id
            device_registry = dr.async_get(self.hass)
            device_entry = device_registry.async_get_device(
                connections={(CONNECTION_BLUETOOTH, address)}
            )
            if device_entry and (
                device_entry.hw_version != last_state.hw_version
                or device_entry.sw_version != last_state.sw_version
            ):
                hw_version = last_state.hw_version
                if hw_version is None or hw_version == "":
                    hw_version = device_entry.hw_version

                sw_version = last_state.sw_version
                if sw_version is None or sw_version == "":
                    sw_version = device_entry.sw_version

                device_registry.async_update_device(
                    device_entry.id,
                    hw_version=hw_version,
                    sw_version=sw_version,
                )
                self._attr_device_info.update(
                    {HW_VERSION_KEY: hw_version, SW_VERSION_KEY: sw_version}
                )
            self._attr_available = True

        await self._coordinator.async_start(self.handle_update)
        _LOGGER.info("Sensor added to Home Assistant: %s", self.entity_id)

    @callback
    def _async_read_entity_options(self) -> None:
        _LOGGER.debug("Reading entity options for sensor: %s", self.entity_id)
        previous_unit = self._sensor_option_unit_of_measurement
        super()._async_read_entity_options()
        if self._sensor_option_unit_of_measurement != previous_unit:
            match self._sensor_option_unit_of_measurement:
                case "kg" | "g" | "mg" | "µg":
                    self._coordinator.set_display_unit(WeightUnit.KG)
                case "lb" | "oz":
                    self._coordinator.set_display_unit(WeightUnit.LB)
                case "st":
                    self._coordinator.set_display_unit(WeightUnit.ST)
                case _:
                    _LOGGER.warning("Unknown unit of measurement")

    def handle_update(
        self,
        data: ScaleData,
    ) -> None:
        """Handle updated data from the scale.

        This method is called when new data is received from the scale.
        It updates the sensor's state and triggers a state update in HA.

        Args:
            data: The new scale data.

        """
        _LOGGER.debug(
            "Received update for sensor %s: %s",
            self.entity_id,
            data.measurements[WEIGHT_KEY],
        )
        self._attr_available = True
        self._attr_native_value = data.measurements[WEIGHT_KEY]

        if (
            self._sensor_option_unit_of_measurement is None
            or self._sensor_option_unit_of_measurement == UNDEFINED
        ):
            match data.display_unit:
                case WeightUnit.KG:
                    self._sensor_option_unit_of_measurement = (
                        UnitOfMass.KILOGRAMS
                    )
                case WeightUnit.LB:
                    self._sensor_option_unit_of_measurement = UnitOfMass.POUNDS
                case WeightUnit.ST:
                    self._sensor_option_unit_of_measurement = UnitOfMass.STONES

        address = self._id
        device_registry = dr.async_get(self.hass)
        device_entry = device_registry.async_get_device(
            connections={(CONNECTION_BLUETOOTH, address)}
        )
        if device_entry and (
            device_entry.hw_version != data.hw_version
            or device_entry.sw_version != data.sw_version
        ):
            hw_version = data.hw_version
            if hw_version is None or hw_version == "":
                hw_version = device_entry.hw_version

            sw_version = data.sw_version
            if sw_version is None or sw_version == "":
                sw_version = device_entry.sw_version

            device_registry.async_update_device(
                device_entry.id, hw_version=hw_version, sw_version=sw_version
            )
            self._attr_device_info.update(
                {HW_VERSION_KEY: hw_version, SW_VERSION_KEY: sw_version}
            )

        self.async_write_ha_state()
        _LOGGER.debug("Sensor %s updated successfully", self.entity_id)

    @property
    def extra_state_attributes(self):
        """Return the state attributes of the sensor."""
        return {
            DISPLAY_UNIT_KEY: self._sensor_option_unit_of_measurement,
            HW_VERSION_KEY: self._attr_device_info.get(HW_VERSION_KEY),
            SW_VERSION_KEY: self._attr_device_info.get(SW_VERSION_KEY),
        }

    @property
    def extra_restore_state_data(self) -> ScaleSensorExtraStoredData:
        """Return sensor specific state data to be restored."""
        return ScaleSensorExtraStoredData(
            self.native_value,
            self.native_unit_of_measurement,
            self._sensor_option_unit_of_measurement,
            self._attr_device_info.get(HW_VERSION_KEY),
            self._attr_device_info.get(SW_VERSION_KEY),
        )

    async def async_get_last_sensor_data(
        self,
    ) -> ScaleSensorExtraStoredData | None:
        """Restore Scale Sensor Extra Stored Data."""
        if (
            restored_last_extra_data := await self.async_get_last_extra_data()
        ) is None:
            return None

        return ScaleSensorExtraStoredData.from_dict(
            restored_last_extra_data.as_dict()
        )
