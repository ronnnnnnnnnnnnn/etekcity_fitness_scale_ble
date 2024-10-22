"""Support for Etekcity Fitness Scale BLE sensors."""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Self

from etekcity_esf551_ble import IMPEDANCE_KEY, WEIGHT_KEY, Sex, WeightUnit
from sensor_state_data import Units

from homeassistant import config_entries
from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntityDescription,
    SensorExtraStoredData,
    SensorStateClass,
    async_update_suggested_units,
)
from homeassistant.const import CONF_UNIT_SYSTEM, UnitOfMass
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_BIRTHDATE, CONF_CALC_BODY_METRICS, CONF_HEIGHT, CONF_SEX, DOMAIN
from .coordinator import ScaleData, ScaleDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

SENSOR_DESCRIPTIONS = [
    SensorEntityDescription(
        key="body_mass_index",
        icon="mdi:human-male-height-variant",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="body_fat_percentage",
        icon="mdi:human-handsdown",
        native_unit_of_measurement=Units.PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="fat_free_weight",
        icon="mdi:run",
        device_class=SensorDeviceClass.WEIGHT,
        native_unit_of_measurement=UnitOfMass.KILOGRAMS,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="subcutaneous_fat_percentage",
        icon="mdi:human-handsdown",
        native_unit_of_measurement=Units.PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="visceral_fat_value",
        icon="mdi:human-handsdown",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="body_water_percentage",
        icon="mdi:water-percent",
        native_unit_of_measurement=Units.PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="basal_metabolic_rate",
        icon="mdi:fire",
        native_unit_of_measurement="cal",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="skeletal_muscle_percentage",
        icon="mdi:weight-lifter",
        native_unit_of_measurement=Units.PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="muscle_mass",
        icon="mdi:weight-lifter",
        device_class=SensorDeviceClass.WEIGHT,
        native_unit_of_measurement=UnitOfMass.KILOGRAMS,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="bone_mass",
        icon="mdi:bone",
        device_class=SensorDeviceClass.WEIGHT,
        native_unit_of_measurement=UnitOfMass.KILOGRAMS,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="protein_percentage",
        icon="mdi:egg-fried",
        native_unit_of_measurement=Units.PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="metabolic_age",
        icon="mdi:human-walker",
        state_class=SensorStateClass.MEASUREMENT,
    ),
]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: config_entries.ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the scale sensors."""
    _LOGGER.debug("Setting up scale sensors for entry: %s", entry.entry_id)
    address = entry.unique_id
    coordinator: ScaleDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [
        ScaleWeightSensor(
            entry.title,
            address,
            coordinator,
            SensorEntityDescription(
                key=WEIGHT_KEY,
                icon="mdi:human-handsdown",
                device_class=SensorDeviceClass.WEIGHT,
                native_unit_of_measurement=UnitOfMass.KILOGRAMS,
                state_class=SensorStateClass.MEASUREMENT,
            ),
        ),
        ScaleSensor(
            entry.title,
            address,
            coordinator,
            SensorEntityDescription(
                key=IMPEDANCE_KEY,
                icon="mdi:omega",
                native_unit_of_measurement=Units.OHM,
                state_class=SensorStateClass.MEASUREMENT,
            ),
        ),
    ]

    if entry.data.get(CONF_CALC_BODY_METRICS):
        sex: Sex = Sex.Male if entry.data.get(CONF_SEX) == "Male" else Sex.Female

        await coordinator.enable_body_metrics(
            sex,
            date.fromisoformat(entry.data.get(CONF_BIRTHDATE)),
            entry.data.get(CONF_HEIGHT) / 100,
        )
        entities += [
            ScaleSensor(entry.title, address, coordinator, desc)
            for desc in SENSOR_DESCRIPTIONS
        ]

    def _update_unit(sensor: ScaleSensor, unit: str) -> ScaleSensor:
        if sensor._attr_device_class == SensorDeviceClass.WEIGHT:
            sensor._attr_suggested_unit_of_measurement = unit
        return sensor

    display_unit: UnitOfMass = entry.data.get(CONF_UNIT_SYSTEM)
    coordinator.set_display_unit(
        WeightUnit.KG if display_unit == UnitOfMass.KILOGRAMS else WeightUnit.LB
    )
    entities = list(
        map(
            lambda sensor: _update_unit(sensor, display_unit),
            entities,
        )
    )
    async_add_entities(entities)
    async_update_suggested_units(hass)
    await coordinator.async_start()
    _LOGGER.debug("Scale sensors setup completed for entry: %s", entry.entry_id)


class ScaleSensor(RestoreSensor):
    """Base sensor implementation for Etekcity scale measurements."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_available = False

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
        self._attr_device_class = entity_description.device_class
        self._attr_state_class = entity_description.state_class
        self._attr_native_unit_of_measurement = (
            entity_description.native_unit_of_measurement
        )
        self._attr_icon = entity_description.icon

        self._attr_name = f"{entity_description.key.replace("_", " ").title()}"

        self._attr_unique_id = f"{name}_{entity_description.key}"

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

        self._attr_available = await self.async_restore_data()

        self.async_on_remove(self._coordinator.add_listener(self.handle_update))
        _LOGGER.info("Sensor added to Home Assistant: %s", self.entity_id)

    async def async_restore_data(self) -> bool:
        """Restore last state from storage."""
        if last_state := await self.async_get_last_sensor_data():
            _LOGGER.debug("Restoring previous state for sensor: %s", self.entity_id)
            self._attr_native_value = last_state.native_value
            return True
        return False

    def handle_update(
        self,
        data: ScaleData,
    ) -> None:
        """Handle updated data from the scale.

        This method is called when new data is received from the scale.
        It updates the sensor's state and triggers a state update in Home Assistant.

        Args:
            data: The new scale data.

        """
        if measurement := data.measurements.get(self.entity_description.key):
            _LOGGER.debug(
                "Received update for sensor %s: %s",
                self.entity_id,
                measurement,
            )
            self._attr_available = True
            self._attr_native_value = measurement

            self.async_write_ha_state()
            _LOGGER.debug("Sensor %s updated successfully", self.entity_id)


HW_VERSION_KEY = "hw_version"
SW_VERSION_KEY = "sw_version"


@dataclass
class ScaleWeightSensorExtraStoredData(SensorExtraStoredData):
    """Object to hold extra stored data for the scale sensor."""

    hw_version: str
    sw_version: str

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation of the scale sensor data."""
        data = super().as_dict()
        data[HW_VERSION_KEY] = self.hw_version
        data[SW_VERSION_KEY] = self.sw_version

        return data

    @classmethod
    def from_dict(cls, restored: dict[str, Any]) -> Self | None:
        """Initialize a stored sensor state from a dict."""
        extra = SensorExtraStoredData.from_dict(restored)
        if extra is None:
            return None

        restored.setdefault("")
        hw_version: str = restored.get(HW_VERSION_KEY)
        sw_version: str = restored.get(SW_VERSION_KEY)

        return cls(
            extra.native_value,
            extra.native_unit_of_measurement,
            hw_version,
            sw_version,
        )


class ScaleWeightSensor(ScaleSensor):
    """Representation of a weight sensor for the Etekcity scale."""

    def __init__(
        self,
        name: str,
        address: str,
        coordinator: ScaleDataUpdateCoordinator,
        entity_description: SensorEntityDescription,
    ) -> None:
        self._id = address
        super().__init__(name, address, coordinator, entity_description)

    async def async_restore_data(self) -> bool:
        """Restore last state from storage."""
        if last_state := await self.async_get_last_sensor_data():
            _LOGGER.debug("Restoring previous state for sensor: %s", self.entity_id)
            self._attr_native_value = last_state.native_value
            self._attr_native_unit_of_measurement = (
                last_state.native_unit_of_measurement
            )

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
                if hw_version == None or hw_version == "":
                    hw_version = device_entry.hw_version

                sw_version = last_state.sw_version
                if sw_version == None or sw_version == "":
                    sw_version = device_entry.sw_version

                device_registry.async_update_device(
                    device_entry.id, hw_version=hw_version, sw_version=sw_version
                )
                self._attr_device_info.update(
                    {HW_VERSION_KEY: hw_version, SW_VERSION_KEY: sw_version}
                )
            return True
        return False

    def handle_update(
        self,
        data: ScaleData,
    ) -> None:
        """Handle updated data from the scale."""

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
            if hw_version == None or hw_version == "":
                hw_version = device_entry.hw_version

            sw_version = data.sw_version
            if sw_version == None or sw_version == "":
                sw_version = device_entry.sw_version

            device_registry.async_update_device(
                device_entry.id, hw_version=hw_version, sw_version=sw_version
            )
            self._attr_device_info.update(
                {HW_VERSION_KEY: hw_version, SW_VERSION_KEY: sw_version}
            )

        super().handle_update(data)

    @property
    def extra_state_attributes(self):
        """Return the state attributes of the sensor."""
        return {
            HW_VERSION_KEY: self._attr_device_info.get(HW_VERSION_KEY),
            SW_VERSION_KEY: self._attr_device_info.get(SW_VERSION_KEY),
        }

    @property
    def extra_restore_state_data(self) -> ScaleWeightSensorExtraStoredData:
        """Return sensor specific state data to be restored."""
        return ScaleWeightSensorExtraStoredData(
            self.native_value,
            self.native_unit_of_measurement,
            self._attr_device_info.get(HW_VERSION_KEY),
            self._attr_device_info.get(SW_VERSION_KEY),
        )

    async def async_get_last_sensor_data(
        self,
    ) -> ScaleWeightSensorExtraStoredData | None:
        """Restore Scale Sensor Extra Stored Data."""
        if (restored_last_extra_data := await self.async_get_last_extra_data()) is None:
            return None

        return ScaleWeightSensorExtraStoredData.from_dict(
            restored_last_extra_data.as_dict()
        )
