"""Support for Etekcity Fitness Scale BLE sensors."""

from dataclasses import dataclass
import logging
from typing import Any, Self

from etekcity_esf551_ble import IMPEDANCE_KEY, WEIGHT_KEY, WeightUnit
from sensor_state_data import Units

from homeassistant import config_entries
from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorExtraStoredData,
    SensorStateClass,
    async_update_suggested_units,
)
from homeassistant.const import CONF_UNIT_SYSTEM, EntityCategory, UnitOfMass
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_BODY_METRICS_ENABLED,
    CONF_CALC_BODY_METRICS,
    CONF_SCALE_DISPLAY_UNIT,
    CONF_USER_ID,
    CONF_USER_NAME,
    CONF_USER_PROFILES,
    DOMAIN,
    get_sensor_unique_id,
)
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

# Diagnostic sensor descriptions
DIAGNOSTIC_SENSOR_DESCRIPTIONS = [
    SensorEntityDescription(
        key="user_directory",
        name="User Directory",
        icon="mdi:account-multiple",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    SensorEntityDescription(
        key="pending_measurements",
        name="Pending Measurements",
        icon="mdi:clipboard-list",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: config_entries.ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the scale sensors with multi-user support."""
    _LOGGER.debug("Setting up scale sensors for entry: %s", entry.entry_id)
    address = entry.unique_id
    coordinator: ScaleDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]

    # Get user profiles from config entry
    user_profiles = entry.data.get(CONF_USER_PROFILES, [])

    # Handle migration from v1 (old single-user format)
    if entry.data.get(CONF_CALC_BODY_METRICS):
        _LOGGER.debug("Migrating from v1 single-user format")
        # This is a v1 entry, set up single user sensors
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
                user_id=None,  # v1 doesn't have user_id
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
                user_id=None,  # v1 doesn't have user_id
            ),
        ]

        entities += [
            ScaleSensor(entry.title, address, coordinator, desc, user_id=None)
            for desc in SENSOR_DESCRIPTIONS
        ]

        display_unit: UnitOfMass = entry.data.get(
            CONF_UNIT_SYSTEM, UnitOfMass.KILOGRAMS
        )
    else:
        # v2 multi-user format
        entities = []

        # Create sensors for each user profile
        for user_profile in user_profiles:
            user_id = user_profile.get(CONF_USER_ID)
            user_name = user_profile.get(CONF_USER_NAME, "User")
            body_metrics_enabled = user_profile.get(CONF_BODY_METRICS_ENABLED, False)

            # Create weight and impedance sensors for this user
            user_entities = [
                ScaleUserWeightSensor(
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
                    user_id=user_id,
                    user_name=user_name,
                ),
                ScaleUserSensor(
                    entry.title,
                    address,
                    coordinator,
                    SensorEntityDescription(
                        key=IMPEDANCE_KEY,
                        icon="mdi:omega",
                        native_unit_of_measurement=Units.OHM,
                        state_class=SensorStateClass.MEASUREMENT,
                    ),
                    user_id=user_id,
                    user_name=user_name,
                ),
            ]

            # Add body metrics sensors if enabled for this user
            if body_metrics_enabled:
                user_entities += [
                    ScaleUserSensor(
                        entry.title,
                        address,
                        coordinator,
                        desc,
                        user_id=user_id,
                        user_name=user_name,
                    )
                    for desc in SENSOR_DESCRIPTIONS
                ]

            entities.extend(user_entities)

        display_unit: UnitOfMass = entry.data.get(
            CONF_SCALE_DISPLAY_UNIT, UnitOfMass.KILOGRAMS
        )

        # Add diagnostic sensors for v2 (multi-user) only
        entities.extend(
            [
                ScaleUserDirectorySensor(entry.title, address, coordinator),
                ScalePendingMeasurementsSensor(entry.title, address, coordinator),
            ]
        )

    # Update suggested units for weight sensors
    def _update_unit(sensor: ScaleSensor, unit: str) -> ScaleSensor:
        if getattr(sensor, "_attr_device_class", None) == SensorDeviceClass.WEIGHT:
            sensor._attr_suggested_unit_of_measurement = unit
        return sensor

    # Set display unit on coordinator
    coordinator.set_display_unit(
        WeightUnit.KG if display_unit == UnitOfMass.KILOGRAMS else WeightUnit.LB
    )

    entities = [_update_unit(sensor, display_unit) for sensor in entities]
    async_add_entities(entities)
    async_update_suggested_units(hass)

    # Start the coordinator
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
        user_id: str | None = None,
    ) -> None:
        """Initialize the scale sensor.

        Args:
            name: The name of the sensor.
            address: The Bluetooth address of the scale.
            coordinator: The data update coordinator for the scale.
            entity_description: Description of the sensor entity.
            user_id: Optional user ID for v1 compatibility.

        """
        self.entity_description = entity_description
        self._attr_device_class = entity_description.device_class
        self._attr_state_class = entity_description.state_class
        self._attr_native_unit_of_measurement = (
            entity_description.native_unit_of_measurement
        )
        self._attr_icon = entity_description.icon

        self._attr_name = f"{entity_description.key.replace("_", " ").title()}"

        # Store user_id for history lookups (empty string for v1 compatibility)
        self._user_id = user_id or ""

        self._attr_unique_id = get_sensor_unique_id(
            name, self._user_id, entity_description.key
        )

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

        # v1 compatibility - register standard listener
        # Subclasses (v2) override this method and register user-specific listeners
        self.async_on_remove(self._coordinator.add_listener(self.handle_update))
        _LOGGER.debug("Sensor added to Home Assistant: %s", self.entity_id)

    async def async_restore_data(self) -> bool:
        """Restore last state from storage."""
        if last_state := await self.async_get_last_sensor_data():
            _LOGGER.debug("Restoring previous state for sensor: %s", self.entity_id)
            self._attr_native_value = last_state.native_value
            return True
        return False

    async def async_get_last_sensor_data(self) -> SensorExtraStoredData | None:
        """Restore Sensor Extra Stored Data."""
        if (restored_last_extra_data := await self.async_get_last_extra_data()) is None:
            return None

        return SensorExtraStoredData.from_dict(restored_last_extra_data.as_dict())

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
        # Update sensor with new data
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
        else:
            # Key is missing from update - mark sensor unavailable
            # Coordinator is the source of truth and decides what goes in measurements
            _LOGGER.debug(
                "Value for sensor %s not in update, marking unavailable",
                self.entity_id,
            )
            self._attr_available = False
            self._attr_native_value = None
            self.async_write_ha_state()


HW_VERSION_KEY = "hw_version"
SW_VERSION_KEY = "sw_version"


@dataclass
class ScaleWeightSensorExtraStoredData(SensorExtraStoredData):
    """Object to hold extra stored data for the scale weight sensor."""

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
    """Representation of a weight sensor for the Etekcity scale (v1 compatibility)."""

    def __init__(
        self,
        name: str,
        address: str,
        coordinator: ScaleDataUpdateCoordinator,
        entity_description: SensorEntityDescription,
        user_id: str | None = None,
    ) -> None:
        self._id = address
        super().__init__(name, address, coordinator, entity_description, user_id)

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
                if hw_version is None or hw_version == "":
                    hw_version = device_entry.hw_version

                sw_version = last_state.sw_version
                if sw_version is None or sw_version == "":
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


class ScaleUserSensor(ScaleSensor):
    """Base sensor for user-specific measurements (v2 multi-user)."""

    def __init__(
        self,
        name: str,
        address: str,
        coordinator: ScaleDataUpdateCoordinator,
        entity_description: SensorEntityDescription,
        user_id: str,
        user_name: str,
    ) -> None:
        """Initialize user-specific sensor.

        Args:
            name: The name of the scale device.
            address: The Bluetooth address of the scale.
            coordinator: The data update coordinator for the scale.
            entity_description: Description of the sensor entity.
            user_id: The unique ID of the user.
            user_name: The name of the user.
        """
        super().__init__(name, address, coordinator, entity_description)

        self._user_id = user_id
        self._user_name = user_name

        # Update entity attributes for user-specific sensor
        # For legacy v1 user (empty user_id), don't prepend user name
        if user_id == "":
            self._attr_name = f"{entity_description.key.replace('_', ' ').title()}"
        else:
            self._attr_name = (
                f"{user_name}'s {entity_description.key.replace('_', ' ').title()}"
            )

        # Use helper function to construct unique_id (handles v1 compatibility)
        self._attr_unique_id = get_sensor_unique_id(
            name, user_id, entity_description.key
        )

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        _LOGGER.debug("Adding user sensor to Home Assistant: %s", self.entity_id)

        # Call RestoreSensor's async_added_to_hass, but not ScaleSensor's
        # to avoid duplicate listener registration
        await RestoreSensor.async_added_to_hass(self)

        # Restore data
        self._attr_available = await self.async_restore_data()

        # Register user-specific callback using direct callback registry
        self.async_on_remove(
            self._coordinator.add_user_listener(self._user_id, self.handle_update)
        )
        _LOGGER.debug("User sensor added to Home Assistant: %s", self.entity_id)


class ScaleUserWeightSensor(ScaleUserSensor):
    """Weight sensor for user-specific measurements (v2 multi-user)."""

    def __init__(
        self,
        name: str,
        address: str,
        coordinator: ScaleDataUpdateCoordinator,
        entity_description: SensorEntityDescription,
        user_id: str,
        user_name: str,
    ) -> None:
        """Initialize user-specific weight sensor."""
        self._id = address
        super().__init__(
            name, address, coordinator, entity_description, user_id, user_name
        )

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
                if hw_version is None or hw_version == "":
                    hw_version = device_entry.hw_version

                sw_version = last_state.sw_version
                if sw_version is None or sw_version == "":
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


class ScaleUserDirectorySensor(SensorEntity):
    """Diagnostic sensor that lists all users with their IDs."""

    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(
        self,
        name: str,
        address: str,
        coordinator: ScaleDataUpdateCoordinator,
    ) -> None:
        """Initialize the user directory sensor.

        Args:
            name: The name of the sensor.
            address: The Bluetooth address of the scale.
            coordinator: The coordinator instance.
        """
        self.entity_description = DIAGNOSTIC_SENSOR_DESCRIPTIONS[0]
        self._attr_unique_id = f"{name}_user_directory"
        self._attr_device_info = DeviceInfo(
            connections={(CONNECTION_BLUETOOTH, address)},
            name=name,
            manufacturer="Etekcity",
        )
        self._coordinator = coordinator

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        # Update immediately when added
        self.async_write_ha_state()

    @property
    def native_value(self) -> int:
        """Return the count of users."""
        return len(self._coordinator.get_user_profiles())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return user directory as attributes."""
        users = []
        for profile in self._coordinator.get_user_profiles():
            user_data = {
                "user_id": profile.get(CONF_USER_ID, ""),
                "name": profile.get(CONF_USER_NAME, ""),
                "has_body_metrics": profile.get(CONF_BODY_METRICS_ENABLED, False),
            }
            users.append(user_data)

        return {"users": users}


class ScalePendingMeasurementsSensor(SensorEntity):
    """Diagnostic sensor that lists pending (unassigned) measurements."""

    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(
        self,
        name: str,
        address: str,
        coordinator: ScaleDataUpdateCoordinator,
    ) -> None:
        """Initialize the pending measurements sensor.

        Args:
            name: The name of the sensor.
            address: The Bluetooth address of the scale.
            coordinator: The coordinator instance.
        """
        self.entity_description = DIAGNOSTIC_SENSOR_DESCRIPTIONS[1]
        self._attr_unique_id = f"{name}_pending_measurements"
        self._attr_device_info = DeviceInfo(
            connections={(CONNECTION_BLUETOOTH, address)},
            name=name,
            manufacturer="Etekcity",
        )
        self._coordinator = coordinator
        self._remove_listener = None

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()

        # Register diagnostic callback for updates (no data passed)
        self._remove_listener = self._coordinator.add_diagnostic_listener(
            self._handle_coordinator_update
        )

        # Update immediately when added
        self.async_write_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from hass."""
        # Unregister callback
        if self._remove_listener:
            self._remove_listener()
        await super().async_will_remove_from_hass()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle coordinator updates.

        This is a diagnostic sensor that pulls data directly from the coordinator,
        so it doesn't need ScaleData passed to it.
        """
        # Update state when coordinator data changes (new measurements)
        self.async_write_ha_state()

    @property
    def native_value(self) -> int:
        """Return the count of pending measurements."""
        return len(self._coordinator.get_pending_measurements())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return pending measurements as attributes."""
        pending_data = []
        for (
            timestamp,
            pending_dict,
        ) in self._coordinator.get_pending_measurements().items():
            raw_measurements = pending_dict["measurements"]
            measurement = {
                "timestamp": timestamp,
                "weight_kg": raw_measurements.get("weight"),
            }
            # Add impedance if available
            if "impedance" in raw_measurements:
                measurement["impedance"] = raw_measurements["impedance"]

            pending_data.append(measurement)

        # Sort by timestamp (most recent first)
        pending_data.sort(key=lambda x: x["timestamp"], reverse=True)

        return {"pending": pending_data}
