from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from etekcity_esf551_ble import (
    EtekcitySmartFitnessScale,
    ScaleData,
    WeightUnit,
)
from homeassistant.core import callback

_LOGGER = logging.getLogger(__name__)


class ScaleDataUpdateCoordinator:
    """Coordinator to manage data updates for a scale device.

    This class handles the communication with the Etekcity Smart Fitness Scale
    and coordinates updates to the Home Assistant entities.
    """

    _client: EtekcitySmartFitnessScale = None
    _display_unit: WeightUnit = None

    def __init__(self, address: str) -> None:
        """Initialize the ScaleDataUpdateCoordinator.

        Args:
            address (str): The Bluetooth address of the scale.

        """
        self.address = address
        self._lock = asyncio.Lock()

    def set_display_unit(self, unit: WeightUnit) -> None:
        """Set the display unit for the scale."""
        _LOGGER.debug("Setting display unit to: %s", unit.name)
        self._display_unit = unit
        if self._client:
            self._client.display_unit = unit

    @callback
    async def async_start(self, update_callback: Callable[[ScaleData], None]) -> None:
        """Start the coordinator and initialize the scale client.

        This method sets up the EtekcitySmartFitnessScale client and starts
        listening for updates from the scale.

        Args:
            update_callback (Callable[[ScaleData], None]): A callback function
                that will be called when new data is received from the scale.

        """
        _LOGGER.debug(
            "Starting ScaleDataUpdateCoordinator for address: %s", self.address
        )
        async with self._lock:
            if self._client:
                _LOGGER.debug("Stopping existing client")
                await self._client.async_stop()
            _LOGGER.debug("Initializing new EtekcitySmartFitnessScale client")
            self._client = EtekcitySmartFitnessScale(
                self.address, update_callback, self._display_unit
            )
            await self._client.async_start()
        _LOGGER.debug("ScaleDataUpdateCoordinator started successfully")

    @callback
    async def async_stop(self) -> None:
        """Stop the coordinator and clean up resources."""
        _LOGGER.debug(
            "Stopping ScaleDataUpdateCoordinator for address: %s", self.address
        )
        async with self._lock:
            if self._client:
                await self._client.async_stop()
                self._client = None
        _LOGGER.debug("ScaleDataUpdateCoordinator stopped successfully")
