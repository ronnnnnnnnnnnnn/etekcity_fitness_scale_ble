"""Config flow for etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import voluptuous as vol
from homeassistant.components.bluetooth import (
    BluetoothServiceInfo,
    async_discovered_service_info,
)
from homeassistant.config_entries import ConfigFlow
from homeassistant.const import CONF_ADDRESS
from homeassistant.data_entry_flow import FlowResult

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


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

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._discovered_devices: dict[str, Discovery] = {}

    async def async_step_bluetooth(
        self, discovery_info: BluetoothServiceInfo
    ) -> FlowResult:
        """Handle the bluetooth discovery step."""
        _LOGGER.debug("Discovered BT device: %s", discovery_info)
        await self.async_set_unique_id(discovery_info.address)
        self._abort_if_unique_id_configured()

        self.context["title_placeholders"] = {"name": title(discovery_info)}

        return await self.async_step_bluetooth_confirm()

    async def async_step_bluetooth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm discovery."""
        if user_input is not None:
            return self.async_create_entry(
                title=self.context["title_placeholders"]["name"], data={}
            )

        self._set_confirm_only()
        return self.async_show_form(
            step_id="bluetooth_confirm",
            description_placeholders=self.context["title_placeholders"],
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

            self.context["title_placeholders"] = {
                "name": discovery.title,
            }

            return self.async_create_entry(title=discovery.title, data={})

        current_addresses = self._async_current_ids()
        for discovery_info in async_discovered_service_info(self.hass):
            address = discovery_info.address
            if (
                address in current_addresses
                or address in self._discovered_devices
            ):
                continue

            if discovery_info.advertisement.local_name is None:
                continue

            if not (
                discovery_info.advertisement.local_name.startswith("Etekcity")
            ):
                continue

            _LOGGER.debug("Found BT Scale")
            _LOGGER.debug("Scale Discovery address: %s", address)
            _LOGGER.debug(
                "Scale Man Data: %s", discovery_info.manufacturer_data
            )
            _LOGGER.debug(
                "Scale advertisement: %s", discovery_info.advertisement
            )
            _LOGGER.debug("Scale device: %s", discovery_info.device)
            _LOGGER.debug(
                "Scale service data: %s", discovery_info.service_data
            )
            _LOGGER.debug(
                "Scale service uuids: %s", discovery_info.service_uuids
            )
            _LOGGER.debug("Scale rssi: %s", discovery_info.rssi)
            _LOGGER.debug(
                "Scale advertisement: %s",
                discovery_info.advertisement.local_name,
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
                }
            ),
        )
