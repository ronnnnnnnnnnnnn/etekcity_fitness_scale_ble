"""Config flow for etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.components.bluetooth import (
    BluetoothServiceInfo,
    async_discovered_service_info,
)
from homeassistant.config_entries import ConfigEntry, ConfigFlow
from homeassistant.const import CONF_ADDRESS, CONF_UNIT_SYSTEM, UnitOfLength, UnitOfMass
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.util.unit_conversion import DistanceConverter

from .const import (
    CONF_BIRTHDATE,
    CONF_CALC_BODY_METRICS,
    CONF_FEET,
    CONF_HEIGHT,
    CONF_INCHES,
    CONF_SCALE_MODEL,
    CONF_SEX,
    DOMAIN,
    SCALE_DETECTION_PATTERNS,
    ScaleModel,
)

_LOGGER = logging.getLogger(__name__)


def detect_scale_model(discovery_info: BluetoothServiceInfo) -> ScaleModel:
    """Detect the scale model based on advertisement data.

    Args:
        discovery_info: The Bluetooth service info for the discovered device.

    Returns:
        The detected scale model.
    """
    # Check for ESF-24 first (more specific pattern)
    esf24_pattern = SCALE_DETECTION_PATTERNS[ScaleModel.ESF24]
    if discovery_info.name and discovery_info.name == esf24_pattern["local_name"]:
        _LOGGER.debug("Detected ESF-24 scale: %s", discovery_info.name)
        return ScaleModel.ESF24

    # Check for ESF-551
    esf551_pattern = SCALE_DETECTION_PATTERNS[ScaleModel.ESF551]
    if discovery_info.name:
        # Use regex pattern matching for ESF-551
        pattern = esf551_pattern.get("local_name_pattern", "").replace("*", ".*")
        if re.search(pattern, discovery_info.name, re.IGNORECASE):
            _LOGGER.debug("Detected ESF-551 scale: %s", discovery_info.name)
            return ScaleModel.ESF551

    # Default to ESF-551 for backward compatibility
    _LOGGER.debug("Unknown scale model, defaulting to ESF-551: %s", discovery_info.name)
    return ScaleModel.ESF551


@dataclasses.dataclass(frozen=True)
class Discovery:
    """Represents a discovered Bluetooth device.

    Attributes:
        title: The name or title of the discovered device.
        discovery_info: Information about the discovered device.

    """

    title: str
    discovery_info: BluetoothServiceInfo


def title(discovery_info: BluetoothServiceInfo) -> str:
    return f"{discovery_info.name} {discovery_info.address}"


class ScaleConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for BT scale."""

    VERSION = 1
    _entry: ConfigEntry

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._discovered_devices: dict[str, Discovery] = {}
        self.metric_schema_dict = {
            vol.Required(
                CONF_SEX,
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=["Male", "Female"],
                    translation_key=CONF_SEX,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(
                CONF_BIRTHDATE,
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.DATE,
                ),
            ),
            vol.Required(CONF_HEIGHT, default=170): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=300,
                    unit_of_measurement=UnitOfLength.CENTIMETERS,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
        }

        self.imperial_schema_dict = {
            vol.Required(
                CONF_SEX,
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=["Male", "Female"],
                    translation_key=CONF_SEX,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(
                CONF_BIRTHDATE,
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.DATE,
                ),
            ),
            vol.Required(CONF_FEET, default=5): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=8,
                    unit_of_measurement=UnitOfLength.FEET,
                    mode=selector.NumberSelectorMode.SLIDER,
                )
            ),
            vol.Required(CONF_INCHES, default=7): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=11,
                    unit_of_measurement=UnitOfLength.INCHES,
                    step=0.5,
                    mode=selector.NumberSelectorMode.SLIDER,
                )
            ),
        }

    async def async_step_bluetooth(
        self, discovery_info: BluetoothServiceInfo
    ) -> FlowResult:
        """Handle the bluetooth discovery step."""
        _LOGGER.debug("Discovered BT device: %s", discovery_info)
        await self.async_set_unique_id(discovery_info.address)
        self._abort_if_unique_id_configured()

        # Detect scale model from advertisement data
        scale_model = detect_scale_model(discovery_info)
        self.context["scale_model"] = scale_model

        self.context["title_placeholders"] = {"name": title(discovery_info)}

        return await self.async_step_bluetooth_confirm()

    async def async_step_bluetooth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm discovery."""
        scale_model = self.context.get("scale_model", ScaleModel.ESF551)
        
        # Handle user input for both scale models
        if user_input is not None:
            if scale_model == ScaleModel.ESF24:
                # ESF24: Create entry with fixed values (no user choice needed)
                return self.async_create_entry(
                    title=self.context["title_placeholders"]["name"],
                    data={
                        CONF_UNIT_SYSTEM: UnitOfMass.KILOGRAMS,  # ESF24 only supports kg
                        CONF_CALC_BODY_METRICS: False,  # ESF24 doesn't support body metrics
                        CONF_SCALE_MODEL: scale_model,
                    },
                )
            else:
                # ESF551: Handle normal configuration flow
                if user_input[CONF_CALC_BODY_METRICS]:
                    self.context[CONF_UNIT_SYSTEM] = user_input[CONF_UNIT_SYSTEM]
                    return await self.async_step_body_metrics()

                return self.async_create_entry(
                    title=self.context["title_placeholders"]["name"],
                    data={
                        CONF_UNIT_SYSTEM: user_input[CONF_UNIT_SYSTEM],
                        CONF_CALC_BODY_METRICS: False,
                        CONF_SCALE_MODEL: scale_model,
                    },
                )

        # Show confirmation form
        description_placeholders = self.context["title_placeholders"].copy()
        
        # For ESF24, show a simple confirmation without unit/body metrics options
        if scale_model == ScaleModel.ESF24:
            description_placeholders["esf24_note"] = " (ESF-24 detected - experimental support, weight only)"
            return self.async_show_form(
                step_id="bluetooth_confirm",
                description_placeholders=description_placeholders,
                data_schema=vol.Schema({}),  # No user input needed for ESF24
            )
        else:
            # For ESF551, show full configuration options
            return self.async_show_form(
                step_id="bluetooth_confirm",
                description_placeholders=description_placeholders,
                data_schema=vol.Schema(
                    {
                        vol.Required(CONF_UNIT_SYSTEM): vol.In(
                            {UnitOfMass.KILOGRAMS: "Metric", UnitOfMass.POUNDS: "Imperial"}
                        ),
                        vol.Required(CONF_CALC_BODY_METRICS, default=False): cv.boolean,
                    }
                ),
            )

    def _convert_height_measurements(
        self, unit: str, user_input: dict[str, Any]
    ) -> tuple[int, int, float]:
        """Convert height measurements between metric and imperial units.

        Args:
            unit: The unit system being used (KILOGRAMS or POUNDS)
            user_input: Dictionary containing height measurements

        Returns:
            tuple: Contains height in (centimeters, feet, inches)

        """
        if unit == UnitOfMass.KILOGRAMS:
            centimeters = user_input[CONF_HEIGHT]
            half_inches = round(centimeters / 1.27)
            feet = half_inches // 24
            inches = (half_inches % 24) / 2
        else:
            feet = user_input[CONF_FEET]
            inches = user_input[CONF_INCHES]
            centimeters = round(
                DistanceConverter.convert(
                    feet,
                    UnitOfLength.FEET,
                    UnitOfLength.CENTIMETERS,
                )
                + DistanceConverter.convert(
                    inches,
                    UnitOfLength.INCHES,
                    UnitOfLength.CENTIMETERS,
                )
            )

        return centimeters, feet, inches

    async def async_step_body_metrics(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        unit = self.context[CONF_UNIT_SYSTEM]
        if user_input is not None:
            centimeters, feet, inches = self._convert_height_measurements(
                unit, user_input
            )
            return self.async_create_entry(
                title=self.context["title_placeholders"]["name"],
                data={
                    CONF_UNIT_SYSTEM: unit,
                    CONF_CALC_BODY_METRICS: True,
                    CONF_SEX: user_input[CONF_SEX],
                    CONF_BIRTHDATE: user_input[CONF_BIRTHDATE],
                    CONF_HEIGHT: centimeters,
                    CONF_FEET: feet,
                    CONF_INCHES: inches,
                    CONF_SCALE_MODEL: self.context.get("scale_model", ScaleModel.ESF551),
                },
            )

        schema = (
            vol.Schema(self.metric_schema_dict)
            if unit == UnitOfMass.KILOGRAMS
            else vol.Schema(self.imperial_schema_dict)
        )

        return self.async_show_form(
            step_id="body_metrics",
            description_placeholders=self.context["title_placeholders"],
            data_schema=schema,
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the user step to pick discovered device."""
        if user_input is not None:
            address = user_input[CONF_ADDRESS]
            await self.async_set_unique_id(address, raise_on_progress=False)
            self._abort_if_unique_id_configured()
            discovery = self._discovered_devices[address]

            # Detect scale model for manual discovery
            scale_model = detect_scale_model(discovery.discovery_info)
            self.context["scale_model"] = scale_model

            if scale_model == ScaleModel.ESF24:
                # ESF24: Create entry with fixed values
                return self.async_create_entry(
                    title=discovery.title,
                    data={
                        CONF_UNIT_SYSTEM: UnitOfMass.KILOGRAMS,
                        CONF_CALC_BODY_METRICS: False,
                        CONF_SCALE_MODEL: scale_model,
                    },
                )
            else:
                # ESF551: Handle normal configuration flow
                if user_input[CONF_CALC_BODY_METRICS]:
                    self.context[CONF_UNIT_SYSTEM] = user_input[CONF_UNIT_SYSTEM]
                    return await self.async_step_body_metrics()

                return self.async_create_entry(
                    title=discovery.title,
                    data={
                        CONF_UNIT_SYSTEM: user_input[CONF_UNIT_SYSTEM],
                        CONF_CALC_BODY_METRICS: False,
                        CONF_SCALE_MODEL: scale_model,
                    },
                )

        current_addresses = self._async_current_ids()
        for discovery_info in async_discovered_service_info(self.hass):
            address = discovery_info.address
            if address in current_addresses or address in self._discovered_devices:
                continue

            _LOGGER.debug("Found BT Scale")
            _LOGGER.debug("Scale Discovery address: %s", address)
            _LOGGER.debug("Scale Man Data: %s", discovery_info.manufacturer_data)
            _LOGGER.debug("Scale advertisement: %s", discovery_info.advertisement)
            _LOGGER.debug("Scale device: %s", discovery_info.device)
            _LOGGER.debug("Scale service data: %s", discovery_info.service_data)
            _LOGGER.debug("Scale service uuids: %s", discovery_info.service_uuids)
            _LOGGER.debug("Scale rssi: %s", discovery_info.rssi)
            _LOGGER.debug(
                "Scale advertisement: %s", discovery_info.advertisement.local_name
            )
            self._discovered_devices[address] = Discovery(
                title(discovery_info), discovery_info
            )

        if not self._discovered_devices:
            return self.async_abort(reason="no_devices_found")

        titles = {
            address: discovery.title
            for (address, discovery) in self._discovered_devices.items()
        }
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_ADDRESS): vol.In(titles),
                    vol.Required(CONF_UNIT_SYSTEM): vol.In(
                        {UnitOfMass.KILOGRAMS: "Metric", UnitOfMass.POUNDS: "Imperial"}
                    ),
                    vol.Required(CONF_CALC_BODY_METRICS, default=False): cv.boolean,
                }
            ),
        )

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None):
        self._entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        if TYPE_CHECKING:
            assert self._entry is not None

        # Get scale model from entry data
        scale_model = self._entry.data.get(CONF_SCALE_MODEL, ScaleModel.ESF551)

        if user_input is not None:
            if scale_model == ScaleModel.ESF24:
                # ESF24: No reconfiguration allowed, just show info
                return self.async_abort(reason="reconfigure_successful")
            
            # ESF551: Handle normal reconfiguration
            if user_input[CONF_CALC_BODY_METRICS]:
                self.context[CONF_UNIT_SYSTEM] = user_input[CONF_UNIT_SYSTEM]
                return await self.async_step_reconfigure_body_metrics()

            return self.async_update_reload_and_abort(
                self._entry,
                title=self._entry.title,
                reason="reconfigure_successful",
                data={CONF_UNIT_SYSTEM: user_input[CONF_UNIT_SYSTEM]},
            )

        # Show reconfiguration form
        if scale_model == ScaleModel.ESF24:
            # ESF24: Show info that reconfiguration is not available
            return self.async_show_form(
                step_id="reconfigure",
                description_placeholders={"esf24_note": " (ESF-24 scale configuration cannot be changed. This device only supports weight measurements in kilograms.)"},
                data_schema=vol.Schema({}),  # No user input needed
            )
        
        # ESF551: Show full reconfiguration options
        if not (body_metrics_enabled := self._entry.data.get(CONF_CALC_BODY_METRICS)):
            body_metrics_enabled = False

        if not (unit_system := self._entry.data.get(CONF_UNIT_SYSTEM)):
            unit_system = vol.UNDEFINED

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_UNIT_SYSTEM, default=unit_system): vol.In(
                        {UnitOfMass.KILOGRAMS: "Metric", UnitOfMass.POUNDS: "Imperial"}
                    ),
                    vol.Required(
                        CONF_CALC_BODY_METRICS, default=body_metrics_enabled
                    ): cv.boolean,
                }
            ),
        )

    async def async_step_reconfigure_body_metrics(
        self, user_input: dict[str, Any] | None = None
    ):
        unit = self.context[CONF_UNIT_SYSTEM]
        if user_input is not None:
            centimeters, feet, inches = self._convert_height_measurements(
                unit, user_input
            )
            return self.async_update_reload_and_abort(
                self._entry,
                title=self._entry.title,
                reason="reconfigure_successful",
                data={
                    CONF_UNIT_SYSTEM: unit,
                    CONF_CALC_BODY_METRICS: True,
                    CONF_SEX: user_input[CONF_SEX],
                    CONF_BIRTHDATE: user_input[CONF_BIRTHDATE],
                    CONF_HEIGHT: centimeters,
                    CONF_FEET: feet,
                    CONF_INCHES: inches,
                },
            )

        schema_dict: dict[vol.Required, Any] = (
            self.metric_schema_dict
            if unit == UnitOfMass.KILOGRAMS
            else self.imperial_schema_dict
        )

        if self._entry.data.get(CONF_CALC_BODY_METRICS):
            schema_dict = {
                vol.Required(
                    key.schema, key.msg, self._entry.data[key.schema], key.description
                ): value
                for key, value in schema_dict.items()
            }

        return self.async_show_form(
            step_id="reconfigure_body_metrics",
            data_schema=vol.Schema(schema_dict),
        )
