"""Coordinator for the etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import date

from etekcity_esf551_ble import (
    EtekcitySmartFitnessScale,
    EtekcitySmartFitnessScaleWithBodyMetrics,
    ScaleData,
    Sex,
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

    body_metrics_enabled: bool = False

    def __init__(self, address: str) -> None:
        """Initialize the ScaleDataUpdateCoordinator.

        Args:
            address (str): The Bluetooth address of the scale.

        """
        self.address = address
        self._lock = asyncio.Lock()
        self._listeners: dict[Callable[[], None], Callable[[ScaleData], None]] = {}

    def set_display_unit(self, unit: WeightUnit) -> None:
        """Set the display unit for the scale."""
        _LOGGER.debug("Setting display unit to: %s", unit.name)
        self._display_unit = unit
        if self._client:
            self._client.display_unit = unit

    async def _async_start(self) -> None:
        if self._client:
            _LOGGER.debug("Stopping existing client")
            await self._client.async_stop()

        if self.body_metrics_enabled:
            _LOGGER.debug(
                "Initializing new EtekcitySmartFitnessScaleWithBodyMetrics client"
            )
            self._client = EtekcitySmartFitnessScaleWithBodyMetrics(
                self.address,
                self.update_listeners,
                self._sex,
                self._birthdate,
                self._height_m,
                self._display_unit,
            )
        else:
            _LOGGER.debug("Initializing new EtekcitySmartFitnessScale client")
            self._client = EtekcitySmartFitnessScale(
                self.address, self.update_listeners, self._display_unit
            )
        await self._client.async_start()

    @callback
    async def async_start(self) -> None:
        """Start the coordinator and initialize the scale client.

        This method sets up the EtekcitySmartFitnessScale client and starts
        listening for updates from the scale.

        """
        _LOGGER.debug(
            "Starting ScaleDataUpdateCoordinator for address: %s", self.address
        )
        async with self._lock:
            await self._async_start()
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

    @callback
    def add_listener(
        self, update_callback: Callable[[ScaleData], None]
    ) -> Callable[[], None]:
        """Listen for data updates."""

        @callback
        def remove_listener() -> None:
            """Remove update listener."""
            self._listeners.pop(remove_listener)

        self._listeners[remove_listener] = update_callback
        return remove_listener

    @callback
    def update_listeners(self, data: ScaleData) -> None:
        """Update all registered listeners."""
        for update_callback in list(self._listeners.values()):
            update_callback(data)

    async def enable_body_metrics(
        self, sex: Sex, birthdate: date, height_m: float
    ) -> None:
        async with self._lock:
            self.body_metrics_enabled = True
            self._sex = sex
            self._birthdate = birthdate
            self._height_m = height_m

            if self._client:
                await self._async_start()

    async def disable_body_metrics(self) -> None:
        async with self._lock:
            if self.body_metrics_enabled:
                self.body_metrics_enabled = False
                self._sex = None
                self._birthdate = None
                self._height_m = None

                if self._client:
                    await self._async_start()
