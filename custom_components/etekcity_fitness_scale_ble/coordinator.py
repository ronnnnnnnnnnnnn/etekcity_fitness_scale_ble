"""Coordinator for the etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from copy import deepcopy
from datetime import datetime, timedelta
import logging
from math import floor
import platform
from collections.abc import Callable
from functools import partial
from typing import Any, Literal
from urllib.parse import quote

from aioesphomeapi import APIClient, BluetoothProxyFeature
from aioesphomeapi.model import BluetoothLEAdvertisement, DeviceInfo
from bleak import BleakError
from bleak.assigned_numbers import AdvertisementDataType
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import (
    AdvertisementData,
    AdvertisementDataCallback,
    BaseBleakScanner,
    get_platform_scanner_backend_type,
)
from bluetooth_data_tools import (
    int_to_bluetooth_address,
    parse_advertisement_data_tuple,
)
from etekcity_esf551_ble import (
    BluetoothScanningMode,
    EtekcitySmartFitnessScale,
    ScaleData,
    WeightUnit,
)
from habluetooth import HaScannerRegistration
from homeassistant.core import HomeAssistant, callback
from homeassistant.components import persistent_notification
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH
from homeassistant.const import UnitOfMass
from homeassistant.util.unit_conversion import MassConverter

from .const import (
    CONF_ENABLE_LIBRARY_LOGGING,
    CONF_HISTORY_RETENTION_DAYS,
    CONF_MAX_HISTORY_SIZE,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_USER_ID,
    CONF_USER_NAME,
    CONF_WEIGHT_HISTORY,
    HISTORY_RETENTION_DAYS,
    MAX_HISTORY_SIZE,
)
from .person_detector import PersonDetector

SYSTEM = platform.system()
IS_LINUX = SYSTEM == "Linux"
IS_MACOS = SYSTEM == "Darwin"


if IS_LINUX:
    from bleak.backends.bluezdbus.advertisement_monitor import OrPattern
    from bleak.backends.bluezdbus.scanner import BlueZScannerArgs

    # or_patterns is a workaround for the fact that passive scanning
    # needs at least one matcher to be set. The below matcher
    # will match all devices.
    PASSIVE_SCANNER_ARGS = BlueZScannerArgs(
        or_patterns=[
            OrPattern(0, AdvertisementDataType.FLAGS, b"\x02"),
            OrPattern(0, AdvertisementDataType.FLAGS, b"\x06"),
            OrPattern(0, AdvertisementDataType.FLAGS, b"\x1a"),
        ]
    )


_LOGGER = logging.getLogger(__name__)


class BluetoothNotAvailableError(Exception):
    """Exception raised when no Bluetooth adapter or ESPHome proxy is available."""

    pass


class BleakScannerESPHome(BaseBleakScanner):
    """
    A BLE scanner implementation that uses ESPHome devices as Bluetooth proxies.

    This scanner connects to one or more ESPHome devices with Bluetooth proxy capability
    and uses them to scan for Bluetooth advertisements. This allows for extended range
    and coverage compared to a single local Bluetooth adapter.
    """

    def __init__(
        self,
        detection_callback: Callable[[BLEDevice, AdvertisementData], None] | None,
        service_uuids: list[str] | None,
        scanning_mode: Literal["active", "passive"],
        clients: list[APIClient],
        **kwargs,
    ):
        """
        Initialize the ESPHome scanner.

        Args:
            detection_callback: Function called when a device advertisement is detected.
            service_uuids: Optional list of service UUIDs to filter advertisements.
            scanning_mode: Whether to use active or passive scanning.
            clients: list of ESPHome API clients to use as Bluetooth proxies.
            **kwargs: Additional arguments (not used).
        """
        super().__init__(detection_callback, service_uuids)

        self._clients = list(clients)
        self._scanning = False

        # Per-client tracking
        self._client_info: dict[APIClient, DeviceInfo | None] = {
            client: None for client in self._clients
        }
        self._client_features: dict[APIClient, int] = {
            client: 0 for client in self._clients
        }
        self._client_unsubscribers: dict[APIClient, Callable[[], None] | None] = {
            client: None for client in self._clients
        }
        self._active_clients: dict[APIClient, dict[str, Any]] = {}

    async def start(self) -> None:
        """Start scanning for devices with enhanced error handling."""
        if self._scanning:
            return

        if not self._clients:
            raise BleakError("No ESPHome clients provided")

        # Track initialization success
        successful_clients = 0

        # Initialize all clients
        for client in self._clients:
            try:
                # Check if client is connected
                if hasattr(client, "is_connected") and not client.is_connected:
                    _LOGGER.warning(
                        "Client %s is not connected, skipping", client.address
                    )
                    continue

                # Get device info with timeout
                try:
                    self._client_info[client] = await asyncio.wait_for(
                        client.device_info(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    _LOGGER.error("Timeout getting device info from %s", client.address)
                    continue

                # Detect Bluetooth features
                self._client_features[client] = self._detect_bluetooth_features(client)

                # Check if the client supports Bluetooth proxy
                supports_proxy = False
                try:
                    supports_proxy = await asyncio.wait_for(
                        self._supports_bluetooth_proxy(client), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    _LOGGER.error(
                        "Timeout checking Bluetooth proxy support for %s",
                        client.address,
                    )
                    continue

                if supports_proxy:
                    self._active_clients[client] = {
                        "name": self._client_info[client].name,
                        "features": self._client_features[client],
                    }

                    # Subscribe to advertisements with error handling
                    try:
                        self._subscribe_to_advertisements(client)
                        successful_clients += 1
                    except Exception as ex:
                        _LOGGER.error(
                            "Failed to subscribe to advertisements for %s: %s",
                            client.address,
                            ex,
                        )
                        continue

                    _LOGGER.debug(
                        "Client %s initialized with features: %s",
                        self._client_info[client].name,
                        self._client_features[client],
                    )
                else:
                    _LOGGER.warning(
                        "Client %s does not support Bluetooth proxy, skipping",
                        client.address,
                    )
            except Exception as ex:
                _LOGGER.warning(
                    "Failed to initialize client %s: %s", client.address, ex
                )

        # Check if we have any active clients
        if not successful_clients:
            raise BleakError(
                "No ESPHome clients support Bluetooth proxy or all initializations failed"
            )

        # Clear the list of seen devices
        self.seen_devices = {}

        self._scanning = True
        _LOGGER.debug(
            "ESPHome scanner started with %d active clients out of %d clients",
            successful_clients,
            len(self._clients),
        )

    async def stop(self) -> None:
        """Stop scanning for devices."""
        if not self._scanning:
            return

        # Unsubscribe from all clients
        for client, unsubscribe in self._client_unsubscribers.items():
            if unsubscribe:
                try:
                    unsubscribe()
                    _LOGGER.debug("Unsubscribed from client %s", client.address)
                except Exception as ex:
                    _LOGGER.warning(
                        "Error unsubscribing from client %s: %s", client.address, ex
                    )
                self._client_unsubscribers[client] = None

        # Clear active clients
        self._active_clients.clear()

        self._scanning = False
        _LOGGER.debug("ESPHome scanner stopped")

    def set_scanning_filter(self, **kwargs) -> None:
        """Set scanning filter for the scanner.

        Note: ESPHome doesn't support additional filters beyond
        the service_uuids provided at initialization.
        """
        # ESPHome doesn't support additional filters
        pass

    def _on_bluetooth_le_advertisement(
        self, client: APIClient, adv: BluetoothLEAdvertisement
    ) -> None:
        """Handle a Bluetooth LE advertisement from a specific client."""
        # Skip if we're filtering by service UUID and this device doesn't match
        if not self.is_allowed_uuid(adv.service_uuids):
            return

        # Create the advertisement data
        advertisement_data = AdvertisementData(
            local_name=adv.name,
            manufacturer_data=adv.manufacturer_data,
            service_data=adv.service_data,
            service_uuids=adv.service_uuids,
            tx_power=adv.tx_power,
            rssi=adv.rssi,
            platform_data=(
                adv,
                client,
            ),  # Store both the advertisement and the source client
        )

        # Update the device in our seen_devices dictionary
        address = int_to_bluetooth_address(adv.address)
        try:
            device = self.create_or_update_device(
                address,
                address,
                adv.name or "",
                adv.manufacturer_data,
                advertisement_data,
            )
        except TypeError:
            # Fallback for older versions of create_or_update_device
            _LOGGER.debug(
                "Using fallback create_or_update_device for bleak version < 1.0.0"
            )
            device = self.create_or_update_device(
                address,
                adv.name or "",
                adv.manufacturer_data,
                advertisement_data,
            )

        # Call the detection callbacks
        self.call_detection_callbacks(device, advertisement_data)

    def _on_bluetooth_le_raw_advertisement(self, client: APIClient, response) -> None:
        """Handle raw Bluetooth LE advertisements from a specific client."""
        if not hasattr(response, "advertisements"):
            _LOGGER.warning(
                "Received raw advertisement response with unknown format from %s: %s",
                client.address,
                response,
            )
            return

        for adv in response.advertisements:
            # Convert the numeric address to a string MAC address
            address = int_to_bluetooth_address(adv.address)
            rssi = adv.rssi

            # Parse the advertisement data using bluetooth_data_tools
            try:
                local_name, service_uuids, service_data, manufacturer_data, tx_power = (
                    parse_advertisement_data_tuple((adv.data,))
                )
            except Exception as ex:
                _LOGGER.debug(
                    "Error parsing advertisement data from %s for %s: %s",
                    client.address,
                    address,
                    ex,
                )
                continue

            # Skip if we're filtering by service UUID and this device doesn't match
            if not self.is_allowed_uuid(service_uuids):
                continue

            # Create the advertisement data
            advertisement_data = AdvertisementData(
                local_name=local_name,
                manufacturer_data=manufacturer_data,
                service_data=service_data,
                service_uuids=service_uuids,
                tx_power=tx_power,
                rssi=rssi,
                platform_data=(
                    adv,
                    client,
                ),  # Store both the advertisement and the source client
            )

            # Update the device in our seen_devices dictionary
            try:
                device = self.create_or_update_device(
                    address,
                    address,
                    local_name or "",
                    manufacturer_data,
                    advertisement_data,
                )
            except TypeError:
                # Fallback for older versions of create_or_update_device
                _LOGGER.debug(
                    "Using fallback create_or_update_device for bleak version < 1.0.0"
                )
                device = self.create_or_update_device(
                    address,
                    local_name or "",
                    manufacturer_data,
                    advertisement_data,
                )

            # Call the detection callbacks
            self.call_detection_callbacks(device, advertisement_data)

    def _subscribe_to_advertisements(self, client: APIClient) -> None:
        """Subscribe to the appropriate advertisement type based on features for a specific client."""
        features = self._client_features[client]

        if features & BluetoothProxyFeature.RAW_ADVERTISEMENTS:
            self._client_unsubscribers[client] = (
                client.subscribe_bluetooth_le_raw_advertisements(
                    partial(self._on_bluetooth_le_raw_advertisement, client)
                )
            )
            _LOGGER.debug("%s: Subscribed to raw advertisements", client.address)
        else:
            self._client_unsubscribers[client] = (
                client.subscribe_bluetooth_le_advertisements(
                    partial(self._on_bluetooth_le_advertisement, client)
                )
            )
            _LOGGER.debug("%s: Subscribed to processed advertisements", client.address)

    def _detect_bluetooth_features(self, client: APIClient) -> int:
        """Detect supported Bluetooth features for a specific client."""
        device_info = self._client_info.get(client)
        if not device_info:
            return 0

        # Check if the device info has the bluetooth_proxy_feature_flags_compat method
        if hasattr(device_info, "bluetooth_proxy_feature_flags_compat"):
            return device_info.bluetooth_proxy_feature_flags_compat(client.api_version)

        # Fallback detection based on features list
        features = device_info.features if device_info else []
        if any("bluetooth" in feature.lower() for feature in features):
            # Assume basic support if "bluetooth" is mentioned in features
            return BluetoothProxyFeature.ACTIVE_CONNECTIONS

        return 0

    async def _supports_bluetooth_proxy(self, client: APIClient) -> bool:
        """Check if a specific ESPHome client supports Bluetooth proxy.

        Args:
            client: The APIClient to check

        Returns:
            bool: True if the client supports Bluetooth proxy, False otherwise.
        """
        # If we've already detected features, use that information
        if self._client_features.get(client, 0) > 0:
            return True

        # Otherwise try to detect features
        if client not in self._client_info or not self._client_info[client]:
            try:
                self._client_info[client] = await client.device_info()
                self._client_features[client] = self._detect_bluetooth_features(client)
                if self._client_features[client] > 0:
                    return True
            except Exception as ex:
                _LOGGER.debug(
                    "Error getting device info for %s: %s", client.address, ex
                )
                return False

        # Fallback to subscription test
        try:
            unsub = client.subscribe_bluetooth_le_advertisements(lambda _: None)
            unsub()
            return True
        except Exception:
            return False


class BleakScannerHybrid(BaseBleakScanner):
    """
    A hybrid BLE scanner that combines native scanning with ESPHome proxies.

    This scanner uses both the local Bluetooth adapter and ESPHome devices with
    Bluetooth proxy capability to scan for advertisements. This provides the best
    coverage by combining local and remote scanning capabilities.
    """

    def __init__(
        self,
        detection_callback: Callable[[BLEDevice, AdvertisementData], None] | None,
        service_uuids: list[str] | None,
        scanning_mode: Literal["active", "passive"],
        clients: list[APIClient],
        adapter: str | None = None,
        **kwargs,
    ):
        """
        Initialize the hybrid scanner.

        Args:
            detection_callback: Function called when a device advertisement is detected.
            service_uuids: Optional list of service UUIDs to filter advertisements.
            scanning_mode: Whether to use active or passive scanning.
            clients: list of ESPHome API clients to use as Bluetooth proxies.
            adapter: The Bluetooth adapter to use for native scanning (Linux only).
            **kwargs: Additional arguments passed to the native scanner.
        """
        super().__init__(None, service_uuids)

        self._native_scanner = None
        self._proxy_scanner = None
        self._scanners: list[BaseBleakScanner] = []
        self._scanning = False

        # Try to create native scanner
        try:
            PlatformBleakScanner = get_platform_scanner_backend_type()
            scanner_kwargs: dict[str, Any] = {
                "bluez": {},
                "cb": {},
            }
            if IS_LINUX:
                # Only Linux supports multiple adapters
                if adapter:
                    scanner_kwargs["adapter"] = adapter
                if scanning_mode == BluetoothScanningMode.PASSIVE:
                    scanner_kwargs["bluez"] = PASSIVE_SCANNER_ARGS
            elif IS_MACOS:
                # We want mac address on macOS
                scanner_kwargs["cb"] = {"use_bdaddr": True}

            self._native_scanner = PlatformBleakScanner(
                detection_callback,
                service_uuids,
                scanning_mode,
                **scanner_kwargs,
            )
            self._scanners.append(self._native_scanner)
            _LOGGER.debug("Native scanner initialized successfully")
        except Exception as ex:
            _LOGGER.warning("Failed to initialize native scanner: %s", ex)

        # Try to create proxy scanner
        try:
            if clients:
                self._proxy_scanner = BleakScannerESPHome(
                    detection_callback, service_uuids, scanning_mode, clients=clients
                )
                self._scanners.append(self._proxy_scanner)
                _LOGGER.debug("Proxy scanner initialized successfully")
            else:
                _LOGGER.warning("No ESPHome clients provided for proxy scanner")
        except Exception as ex:
            _LOGGER.warning("Failed to initialize proxy scanner: %s", ex)

        # Check if we have at least one scanner
        if not self._scanners:
            raise BleakError("Failed to initialize any scanner (native or proxy)")

        self.seen_devices = {}

    async def start(self) -> None:
        """Start scanning for devices."""
        if self._scanning:
            return

        if not self._scanners:
            raise BleakError("No scanners available")

        try:
            # Start all scanners concurrently using asyncio.gather
            await asyncio.gather(*[scanner.start() for scanner in self._scanners])

            # Check if at least one scanner started
            if all(not getattr(s, "_scanning", False) for s in self._scanners):
                raise BleakError("Failed to start any scanner")

            self._scanning = True
            _LOGGER.debug(
                "Hybrid scanner started with %s and %s",
                "native scanner"
                if self._native_scanner in self._scanners
                else "no native scanner",
                "proxy scanner"
                if self._proxy_scanner in self._scanners
                else "no proxy scanner",
            )
        except Exception as ex:
            _LOGGER.exception("Error starting hybrid scanner: %s", ex)
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop scanning for devices."""
        if not self._scanning:
            return

        for scanner in self._scanners:
            try:
                await scanner.stop()
                _LOGGER.debug(f"Stopped scanner: {type(scanner).__name__}")
            except Exception as ex:
                _LOGGER.warning(f"Error stopping {type(scanner).__name__}: {ex}")

        self._scanning = False
        _LOGGER.debug("Hybrid scanner stopped")

    def set_scanning_filter(self, **kwargs) -> None:
        """Set scanning filter for the scanner."""
        for scanner in self._scanners:
            try:
                scanner.set_scanning_filter(**kwargs)
            except Exception as ex:
                _LOGGER.warning(
                    f"Error setting filter on {type(scanner).__name__}: {ex}"
                )

    def register_detection_callback(
        self, callback: AdvertisementDataCallback | None
    ) -> Callable[[], None]:
        for scanner in self._scanners:
            try:
                scanner.register_detection_callback(callback)
            except Exception as ex:
                _LOGGER.exception(
                    f"Error registering detection callback on {type(scanner).__name__}: {ex}"
                )

    @property
    def seen_devices(self) -> dict[str, tuple[BLEDevice, AdvertisementData]]:
        """Get the dictionary of seen devices."""
        seen: dict[str, tuple[BLEDevice, AdvertisementData]] = {}

        for scanner in self._scanners:
            seen |= scanner.seen_devices

        return seen

    @seen_devices.setter
    def seen_devices(
        self, value: dict[str, tuple[BLEDevice, AdvertisementData]]
    ) -> None:
        """Set the dictionary of seen devices."""
        # This is intentionally a no-op as we don't want to override
        # the seen devices of individual scanners
        pass


class ScaleDataUpdateCoordinator:
    """
    Coordinator to manage data updates for a scale device.

    This class handles the communication with the Etekcity Smart Fitness Scale
    and coordinates updates to the Home Assistant entities. Supports multi-user
    detection and routing.
    """

    # Class constants
    MAX_PENDING_MEASUREMENTS = (
        10  # Maximum number of pending (ambiguous) measurements to track
    )

    _client: EtekcitySmartFitnessScale | None = None
    _display_unit: WeightUnit | None = None
    _scanner_change_cb_unregister: Callable[[], None] | None = None

    def __init__(
        self,
        hass: HomeAssistant,
        address: str,
        user_profiles: list[dict],
        device_name: str,
    ) -> None:
        """Initialize the ScaleDataUpdateCoordinator.

        Args:
            hass: The Home Assistant instance.
            address: The Bluetooth address of the scale.
            user_profiles: list of user profile dictionaries.
            device_name: The device name used for entity ID construction.
        """
        self.address = address
        self._hass = hass
        self._device_name = device_name
        self._lock = asyncio.Lock()
        self._listeners: dict[Callable[[], None], Callable[[ScaleData], None]] = {}
        # Diagnostic-only listeners that don't receive scale data (just notifications to refresh)
        self._diagnostic_listeners: list[Callable[[], None]] = []
        # User-specific callback registry: user_id -> list of callbacks
        self._user_callbacks: dict[str, list[Callable[[ScaleData], None]]] = {}
        self._user_profiles = deepcopy(user_profiles)
        self._user_profiles_by_id: dict[str, dict] = {}
        v1_legacy_count = 0  # Track number of v1 legacy users (empty string user_id)

        for profile in self._user_profiles:
            user_id = profile.get(CONF_USER_ID)
            if user_id is not None:
                self._user_profiles_by_id[user_id] = profile
                # Count v1 legacy users (empty string reserved for v1 compatibility)
                if user_id == "":
                    v1_legacy_count += 1
                history = profile.get(CONF_WEIGHT_HISTORY, [])
                user_name = profile.get(CONF_USER_NAME, user_id)
                _LOGGER.debug(
                    "Loaded history for user %s (%s): %d measurements",
                    user_name,
                    user_id,
                    len(history),
                )
                if history:
                    _LOGGER.debug(
                        "   Range: %s to %s",
                        history[0].get("timestamp", "?"),
                        history[-1].get("timestamp", "?"),
                    )
            else:
                _LOGGER.warning(
                    "Skipping user profile without user_id: %s",
                    profile.get(CONF_USER_NAME, "Unknown"),
                )

        # V1 compatibility assertion: Only ONE user can have empty string user_id
        # This preserves entity IDs during v1→v2 migration
        if v1_legacy_count > 1:
            raise ValueError(
                f"Invalid configuration: Found {v1_legacy_count} users with empty string user_id. "
                "Only one user can have empty string user_id (reserved for v1 compatibility). "
                "This indicates corrupted migration data."
            )

        self._person_detector = PersonDetector(hass)
        # Pending measurements awaiting manual assignment
        # Structure: {timestamp: dict with keys:
        #   - "measurements": raw_measurements_dict (weight, impedance)
        #   - "candidates": list of candidate user_ids
        #   - "notified_mobile_services": list of (user_id, service_name) tuples
        # }
        self._pending_measurements: dict[str, dict] = {}
        self._ambiguous_notifications: set[str] = (
            set()
        )  # active notification timestamps
        # Config entry reference for persistence
        self._config_entry_id: str | None = None

    def set_display_unit(self, unit: WeightUnit) -> None:
        """Set the display unit for the scale.

        Args:
            unit: The weight unit to display on the scale.
        """
        _LOGGER.debug("Setting display unit to: %s", unit.name)
        self._display_unit = unit
        if self._client:
            self._client.display_unit = unit

    def get_display_unit(self) -> WeightUnit:
        """Get the current display unit for the scale.

        Returns:
            The weight unit currently configured for display (defaults to KG).
        """
        return self._display_unit if self._display_unit is not None else WeightUnit.KG

    def set_config_entry_id(self, config_entry_id: str) -> None:
        """Set the config entry ID for persistence.

        Args:
            config_entry_id: The config entry ID to store.
        """
        self._config_entry_id = config_entry_id

    def _normalize_measurement(self, measurement: dict) -> dict:
        """Normalize measurement dict to have consistent field order.

        Args:
            measurement: Raw measurement dict.

        Returns:
            Normalized measurement dict with fields in consistent order:
            timestamp, weight_kg, impedance_ohm (if present).
        """
        normalized = {
            "timestamp": measurement["timestamp"],
            "weight_kg": measurement["weight_kg"],
        }
        if "impedance_ohm" in measurement:
            normalized["impedance_ohm"] = measurement["impedance_ohm"]
        return normalized

    def get_user_history(self, user_id: str) -> list[dict]:
        """Get weight history for a user.

        Args:
            user_id: The user ID to get history for.

        Returns:
            List of measurement dicts with 'timestamp', 'weight_kg', and optionally 'impedance_ohm'.
            All measurements have consistent field order.
            Returns empty list if user not found or has no history.
        """
        user_profile = self._user_profiles_by_id.get(user_id)
        if not user_profile:
            return []
        history = user_profile.get(CONF_WEIGHT_HISTORY, [])
        # Normalize all measurements to ensure consistent field order
        return [self._normalize_measurement(m) for m in history]

    def get_user_history_for_display(self, user_id: str) -> list[dict]:
        """Get weight history formatted for display with user-friendly keys.

        Converts weight to display unit and uses friendly key names.

        Args:
            user_id: The user ID to get history for.

        Returns:
            List of measurement dicts formatted for display with keys:
            - "Timestamp" (instead of "timestamp")
            - "Weight (kg)" or "Weight (lbs)" (instead of "weight_kg"/"weight_lb")
            - "Impedance (Ω)" (instead of "impedance_ohm")
        """
        from homeassistant.util.unit_conversion import MassConverter
        from homeassistant.const import UnitOfMass

        history = self.get_user_history(user_id)
        display_unit = self.get_display_unit()
        is_pounds = display_unit == WeightUnit.LB

        display_history = []
        for measurement in history:
            display_measurement = {}
            # Timestamp with friendly key
            display_measurement["Timestamp"] = measurement["timestamp"]

            # Weight with friendly key and unit conversion if needed
            weight_kg = measurement["weight_kg"]
            if is_pounds:
                weight_lb = MassConverter.convert(
                    weight_kg, UnitOfMass.KILOGRAMS, UnitOfMass.POUNDS
                )
                display_measurement["Weight (lbs)"] = round(weight_lb, 2)
            else:
                display_measurement["Weight (kg)"] = round(weight_kg, 2)

            # Impedance with friendly key
            if "impedance_ohm" in measurement:
                display_measurement["Impedance (Ω)"] = measurement["impedance_ohm"]

            display_history.append(display_measurement)

        return display_history

    def get_last_measurement(self, user_id: str) -> dict | None:
        """Get user's last measurement from history.

        Args:
            user_id: The user ID to get last measurement for.

        Returns:
            Last measurement dict or None if no history.
        """
        history = self.get_user_history(user_id)
        result = history[-1] if history else None
        _LOGGER.debug(
            "get_last_measurement(%s): history_size=%d, returning %s",
            user_id,
            len(history),
            "measurement" if result else "None",
        )
        return result

    def get_previous_measurement(self, user_id: str) -> dict | None:
        """Get user's second-to-last measurement from history.

        Args:
            user_id: The user ID to get previous measurement for.

        Returns:
            Previous measurement dict or None if less than 2 measurements.
        """
        history = self.get_user_history(user_id)
        return history[-2] if len(history) >= 2 else None

    def _add_measurement_to_history(
        self,
        user_id: str,
        timestamp: str,
        weight_kg: float,
        impedance_ohm: float | None = None,
    ) -> None:
        """Add measurement to user's history with cleanup.

        Atomic operation: makes all changes to in-memory structures,
        then caller must persist with _update_config_entry().

        Args:
            user_id: User ID to add measurement to.
            timestamp: ISO format timestamp from scale.
            weight_kg: Weight in kilograms.
            impedance_ohm: Optional impedance in ohms.
        """
        user_profile = self._user_profiles_by_id.get(user_id)
        if not user_profile:
            _LOGGER.error(
                "User profile not found for user_id: %s (cannot add measurement)",
                user_id,
            )
            return

        history = user_profile.setdefault(CONF_WEIGHT_HISTORY, [])

        # Check for duplicate timestamp
        if any(m["timestamp"] == timestamp for m in history):
            _LOGGER.warning(
                "Measurement with timestamp %s already exists for user %s, skipping duplicate",
                timestamp,
                user_id,
            )
            return

        # Build measurement dict
        measurement = {"timestamp": timestamp, "weight_kg": weight_kg}
        if impedance_ohm is not None:
            measurement["impedance_ohm"] = impedance_ohm

        # Insert in sorted order using bisect for efficient insertion
        # Find insertion point by comparing timestamps (works on sorted lists)
        if history:
            # Extract timestamps for bisect (assumes history is already sorted)
            timestamps = [m["timestamp"] for m in history]
            insert_pos = bisect_left(timestamps, timestamp)
            history.insert(insert_pos, measurement)
        else:
            # Empty history, just append
            history.append(measurement)

        # Ensure list remains sorted (defensive check in case history wasn't sorted initially)
        # This is O(n log n) but only runs if needed, and history is small (max 100 items)
        if len(history) > 1:
            # Check if actually sorted
            is_sorted = all(
                history[i]["timestamp"] <= history[i + 1]["timestamp"]
                for i in range(len(history) - 1)
            )
            if not is_sorted:
                history.sort(key=lambda m: m["timestamp"])

        # Cleanup old and excess measurements
        self._cleanup_history(user_profile)

        _LOGGER.debug(
            "Added measurement to user %s history: weight=%.2f kg, timestamp=%s (history_size=%d)",
            user_id,
            weight_kg,
            timestamp,
            len(history),
        )

        self._log_user_history(user_id, "after adding measurement")

    def _get_history_retention_days(self) -> int:
        """Get history retention days from config entry or use default.

        Returns:
            Number of days to retain history (default: HISTORY_RETENTION_DAYS).
        """
        if not self._config_entry_id:
            return HISTORY_RETENTION_DAYS

        entry = self._hass.config_entries.async_get_entry(self._config_entry_id)
        if not entry:
            return HISTORY_RETENTION_DAYS

        return entry.data.get(CONF_HISTORY_RETENTION_DAYS, HISTORY_RETENTION_DAYS)

    def _get_max_history_size(self) -> int:
        """Get max history size from config entry or use default.

        Returns:
            Maximum number of measurements per user (default: MAX_HISTORY_SIZE).
        """
        if not self._config_entry_id:
            return MAX_HISTORY_SIZE

        entry = self._hass.config_entries.async_get_entry(self._config_entry_id)
        if not entry:
            return MAX_HISTORY_SIZE

        return entry.data.get(CONF_MAX_HISTORY_SIZE, MAX_HISTORY_SIZE)

    def _is_library_logging_enabled(self) -> bool:
        """Check if library logging is enabled in config entry.

        Returns:
            True if library logging is enabled, False otherwise (default).
        """
        if not self._config_entry_id:
            return False

        entry = self._hass.config_entries.async_get_entry(self._config_entry_id)
        if not entry:
            return False

        return entry.data.get(CONF_ENABLE_LIBRARY_LOGGING, False)

    def _configure_library_logger(self) -> logging.Logger | None:
        """Configure the etekcity_esf551_ble logger based on the advanced setting.

        Returns:
            A child logger to pass to the library when logging is enabled, otherwise None.
        """
        library_root = logging.getLogger("etekcity_esf551_ble")

        if self._is_library_logging_enabled():
            # Re-enable and allow propagation so logs follow HA’s configured level.
            library_root.disabled = False
            library_root.propagate = True
            return _LOGGER.getChild("etekcity_esf551_ble")

        # Disable library logging entirely when the option is off.
        library_root.disabled = True
        library_root.propagate = False
        return None

    def _cleanup_history(self, user_profile: dict) -> None:
        """Remove old and excess measurements from history.

        Enforces configurable HISTORY_RETENTION_DAYS and MAX_HISTORY_SIZE limits.

        Args:
            user_profile: User profile dict to cleanup.
        """
        history = user_profile.get(CONF_WEIGHT_HISTORY, [])

        if not history:
            return

        # Get configurable limits
        retention_days = self._get_history_retention_days()
        max_size = self._get_max_history_size()

        # Remove measurements older than retention window
        # Handle invalid timestamps gracefully to prevent crashes from corrupted data
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        valid_measurements = []
        for m in history:
            timestamp_str = m.get("timestamp")
            if not timestamp_str:
                _LOGGER.warning("Measurement missing timestamp, removing: %s", m)
                continue
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp_str)
                if parsed_timestamp >= cutoff_time:
                    valid_measurements.append(m)
            except (ValueError, TypeError) as ex:
                _LOGGER.warning(
                    "Invalid timestamp format '%s' in measurement, removing: %s",
                    timestamp_str,
                    ex,
                )
                continue

        history[:] = valid_measurements

        # Enforce max size (keep newest)
        if len(history) > max_size:
            history[:] = history[-max_size:]

    def _log_user_history(self, user_id: str, context: str) -> None:
        """Log concise history summary for debugging without spamming logs."""
        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return

        user_profile = self._user_profiles_by_id.get(user_id)
        if not user_profile:
            _LOGGER.debug(
                "History summary skipped: user %s missing (%s)", user_id, context
            )
            return

        history = user_profile.get(CONF_WEIGHT_HISTORY, [])
        _LOGGER.debug(
            "History summary for user %s (%s): %d measurement(s)",
            user_profile.get(CONF_USER_NAME, user_id),
            context,
            len(history),
        )

    def _update_config_entry(self) -> None:
        """Update config entry with current user profiles.

        Persists all in-memory changes to config entry storage.
        """
        if not self._config_entry_id:
            _LOGGER.warning("Cannot update config entry: config entry ID not set")
            return

        entry = self._hass.config_entries.async_get_entry(self._config_entry_id)
        if not entry:
            _LOGGER.error(
                "Config entry not found: entry_id=%s (cannot update user profiles)",
                self._config_entry_id,
            )
            return

        # Persist profiles
        new_data = {**entry.data}
        # Persist a deep copy so config entry data doesn't share references
        new_data["user_profiles"] = deepcopy(self._user_profiles)

        self._hass.config_entries.async_update_entry(entry, data=new_data)
        _LOGGER.debug("Updated config entry with current user profiles")

    async def _get_bluetooth_scanner(self) -> BaseBleakScanner | None:
        """Get the optimal Bluetooth scanner based on available resources.

        Returns:
            A configured Bluetooth scanner or None if no scanner could be created.
        """
        try:
            manager = self._hass.data.get("bluetooth_manager")
            if not manager:
                _LOGGER.debug("Bluetooth manager not available")
                return None

            # Get Bluetooth sources
            sources = manager._sources
            native = False

            # Check for native adapters with better error handling
            try:
                for adapter in manager._bluetooth_adapters.adapters.values():
                    if sources.get(adapter["address"]) is not None:
                        native = True
                        _LOGGER.debug("Found native Bluetooth adapter: %s", adapter)
                        break
                if not native:
                    for adapter in manager._bluetooth_adapters.adapters.keys():
                        if sources.get(adapter) is not None:
                            native = True
                            _LOGGER.debug("Found native Bluetooth adapter: %s", adapter)
                            break
            except (AttributeError, KeyError) as err:
                _LOGGER.warning("Error checking native Bluetooth adapters: %s", err)
                native = False

            # Get ESPHome proxies with error handling
            esphome_clients: list[APIClient] = []
            try:
                proxies = [
                    item.data["source"]
                    for item in self._hass.config_entries.async_entries("bluetooth")
                    if item.data.get("source_domain") == "esphome"
                ]
                esphome_clients = [
                    sources.get(s).connector.client.keywords["client_data"].client
                    for s in proxies
                    if sources.get(s)
                ]
                _LOGGER.debug(
                    "Found %d ESPHome Bluetooth proxies", len(esphome_clients)
                )
            except (AttributeError, KeyError, TypeError) as err:
                _LOGGER.warning("Error getting ESPHome clients: %s", err)
                esphome_clients = []

            # Initialize scanner with error handling
            scanner: BaseBleakScanner | None = None
            if len(esphome_clients) > 0:
                try:
                    if native:
                        scanner = BleakScannerHybrid(
                            None,
                            None,
                            BluetoothScanningMode.PASSIVE,
                            esphome_clients,
                        )
                        _LOGGER.debug(
                            "Created hybrid scanner with native and proxy support"
                        )
                    else:
                        scanner = BleakScannerESPHome(
                            None,
                            None,
                            BluetoothScanningMode.PASSIVE,
                            esphome_clients,
                        )
                        _LOGGER.debug("Created ESPHome proxy scanner")
                except BleakError as err:
                    _LOGGER.warning("Failed to initialize Bluetooth scanner: %s", err)
                    scanner = None
                except Exception as ex:
                    _LOGGER.exception(
                        "Unexpected error creating Bluetooth scanner: %s", ex
                    )
                    scanner = None
            elif not native:
                # No ESPHome proxies AND no native adapter = no Bluetooth available
                raise BluetoothNotAvailableError(
                    "No Bluetooth adapter or ESPHome proxy available"
                )

            return scanner
        except BluetoothNotAvailableError:
            # Re-raise to be handled by caller
            raise
        except Exception as ex:
            _LOGGER.exception("Error getting Bluetooth scanner: %s", ex)
            return None

    async def _async_start(self) -> None:
        """Initialize and start the scale client with improved error handling."""
        try:
            if self._client:
                _LOGGER.debug("Stopping existing client")
                try:
                    await self._client.async_stop()
                except Exception as ex:
                    _LOGGER.warning("Error stopping existing client: %s", ex)
                finally:
                    self._client = None

            # Get the optimal scanner
            try:
                scanner = await self._get_bluetooth_scanner()
            except BluetoothNotAvailableError as err:
                # No Bluetooth adapter or ESPHome proxy available yet
                # This is expected during startup. The _registration_changed
                # callback will restart the client when Bluetooth becomes available.
                _LOGGER.warning(
                    "Bluetooth not available. "
                    "Waiting for a Bluetooth adapter or ESPHome Bluetooth proxy to become available.",
                )
                return  # Graceful exit, callback will retry

            # Initialize client (always use basic client, body metrics calculated per-user)
            try:
                _LOGGER.debug("Initializing new EtekcitySmartFitnessScale client")

                # Conditionally pass logger based on advanced settings
                library_logger = self._configure_library_logger()
                if library_logger:
                    _LOGGER.debug("Library logging enabled, passing child logger")

                self._client = EtekcitySmartFitnessScale(
                    self.address,
                    self.update_listeners,
                    self._display_unit,
                    scanning_mode=BluetoothScanningMode.PASSIVE,
                    bleak_scanner_backend=scanner,
                    logger=library_logger,
                )

                await asyncio.wait_for(self._client.async_start(), timeout=30.0)
                _LOGGER.debug("Scale client started successfully")
            except asyncio.TimeoutError:
                _LOGGER.error(
                    "Timeout while starting scale client for %s", self.address
                )
                if self._client:
                    try:
                        await self._client.async_stop()
                    except Exception:
                        pass
                    self._client = None
                raise
            except BleakError as err:
                _LOGGER.error(
                    "Failed to connect to scale (address: %s, error: %s)",
                    self.address,
                    err,
                )
                if self._client:
                    try:
                        await self._client.async_stop()
                    except Exception:
                        pass
                    self._client = None
                raise
            except Exception as ex:
                _LOGGER.exception("Unexpected error starting scale client: %s", ex)
                if self._client:
                    try:
                        await self._client.async_stop()
                    except Exception:
                        pass
                    self._client = None
                raise
        except Exception as ex:
            _LOGGER.exception("Failed to initialize scale client: %s", ex)
            raise

    def _registration_changed(self, registration: HaScannerRegistration) -> None:
        """Handle Bluetooth scanner registration changes."""
        self._hass.async_create_task(self._async_registration_changed())

    async def _async_registration_changed(self) -> None:
        """Handle Bluetooth scanner registration changes asynchronously."""
        _LOGGER.debug("Bluetooth scanner registration changed, restarting client")
        try:
            async with self._lock:
                await self._async_start()
        except Exception as ex:
            _LOGGER.error(
                "Failed to restart client after scanner registration change: %s", ex
            )

    @callback
    async def async_start(self) -> None:
        """Start the coordinator and initialize the scale client.

        This method sets up the EtekcitySmartFitnessScale client and starts
        listening for updates from the scale.
        """
        _LOGGER.debug(
            "Starting ScaleDataUpdateCoordinator for address: %s", self.address
        )

        # Clean up any existing registration callback
        if self._scanner_change_cb_unregister:
            self._scanner_change_cb_unregister()
            self._scanner_change_cb_unregister = None

        # Register for scanner changes
        bluetooth_manager = self._hass.data.get("bluetooth_manager")
        if bluetooth_manager:
            self._scanner_change_cb_unregister = (
                bluetooth_manager.async_register_scanner_registration_callback(
                    self._registration_changed, None
                )
            )

        async with self._lock:
            try:
                await self._async_start()
                _LOGGER.debug("ScaleDataUpdateCoordinator started successfully")
            except Exception as ex:
                _LOGGER.error(
                    "Failed to start ScaleDataUpdateCoordinator (%s: %s)",
                    type(ex).__name__,
                    ex,
                )
                # Clean up resources on failure
                if self._scanner_change_cb_unregister:
                    self._scanner_change_cb_unregister()
                    self._scanner_change_cb_unregister = None
                raise

    @callback
    async def async_stop(self) -> None:
        """Stop the coordinator and clean up resources."""
        _LOGGER.debug(
            "Stopping ScaleDataUpdateCoordinator for address: %s", self.address
        )
        async with self._lock:
            # Clean up scanner registration callback
            if self._scanner_change_cb_unregister:
                try:
                    self._scanner_change_cb_unregister()
                except Exception as ex:
                    _LOGGER.warning("Error unregistering scanner callback: %s", ex)
                finally:
                    self._scanner_change_cb_unregister = None

            # Stop the client
            if self._client:
                try:
                    await self._client.async_stop()
                except Exception as ex:
                    _LOGGER.warning("Error stopping client: %s", ex)
                finally:
                    self._client = None

            # Clear all pending measurement notifications (they won't persist across reload)
            await self._clear_all_pending_notifications()

        _LOGGER.debug("ScaleDataUpdateCoordinator stopped successfully")

    async def _clear_all_pending_notifications(self) -> None:
        """Clear all notifications for pending measurements.

        Called during unload/stop since pending measurements don't persist
        across reloads, so their notifications should be cleaned up.
        """
        if not self._pending_measurements:
            return

        _LOGGER.debug(
            "Clearing notifications for %d pending measurements",
            len(self._pending_measurements),
        )

        for timestamp, pending_data in self._pending_measurements.items():
            # Dismiss the persistent notification
            try:
                persistent_notification.dismiss(
                    self._hass,
                    notification_id=f"etekcity_scale_{self.address}_{timestamp}",
                )
            except Exception as ex:
                _LOGGER.warning(
                    "Error dismissing persistent notification for %s: %s",
                    timestamp,
                    ex,
                )

            # Dismiss all mobile notifications for this measurement
            notified_services = pending_data.get("notified_mobile_services", [])
            tag = f"scale_measurement_{timestamp}"
            for user_id, service_name in notified_services:
                try:
                    await self._hass.services.async_call(
                        "notify",
                        service_name,
                        {"message": "clear_notification", "data": {"tag": tag}},
                    )
                    _LOGGER.debug(
                        "Dismissed mobile notification for user %s on %s (tag: %s)",
                        user_id,
                        service_name,
                        tag,
                    )
                except Exception as ex:
                    _LOGGER.warning(
                        "Error dismissing mobile notification for %s on %s: %s",
                        user_id,
                        service_name,
                        ex,
                    )

        # Clear the pending measurements dict
        self._pending_measurements.clear()
        self._ambiguous_notifications.clear()

    @callback
    def add_listener(
        self, update_callback: Callable[[ScaleData], None]
    ) -> Callable[[], None]:
        """Listen for data updates.

        Args:
            update_callback: Function to call when new data is received.

        Returns:
            Function to call to remove the listener.
        """

        @callback
        def remove_listener() -> None:
            """Remove update listener."""
            self._listeners.pop(remove_listener, None)

        self._listeners[remove_listener] = update_callback
        return remove_listener

    @callback
    def add_diagnostic_listener(
        self, update_callback: Callable[[], None]
    ) -> Callable[[], None]:
        """Register a diagnostic sensor listener.

        Diagnostic listeners receive simple notifications to refresh their state
        without receiving ScaleData. This prevents unintended side effects on
        other sensors when diagnostic state changes.

        Args:
            update_callback: Function to call when diagnostic state changes (no args).

        Returns:
            Function to call to remove the listener.
        """

        @callback
        def remove_listener() -> None:
            """Remove diagnostic listener."""
            if update_callback in self._diagnostic_listeners:
                self._diagnostic_listeners.remove(update_callback)

        self._diagnostic_listeners.append(update_callback)
        return remove_listener

    def add_user_listener(
        self, user_id: str, update_callback: Callable[[ScaleData], None]
    ) -> Callable[[], None]:
        """Register a callback for a specific user's measurements.

        Args:
            user_id: The user ID this callback is for.
            update_callback: Function to call when new data is received for this user.

        Returns:
            Function to call to remove the listener.
        """
        if user_id not in self._user_callbacks:
            self._user_callbacks[user_id] = []

        self._user_callbacks[user_id].append(update_callback)

        @callback
        def remove_listener() -> None:
            """Remove this user-specific listener."""
            if user_id in self._user_callbacks:
                try:
                    self._user_callbacks[user_id].remove(update_callback)
                    # Clean up empty lists
                    if not self._user_callbacks[user_id]:
                        del self._user_callbacks[user_id]
                except ValueError:
                    pass  # Callback already removed

        return remove_listener

    def _extract_raw_measurements(self, data: ScaleData) -> dict:
        """Extract only raw measurements (not calculated body metrics) from scale data.

        Raw measurements are those that come directly from the scale hardware.
        Body metrics are calculated and depend on user profile, so they should
        not be stored before user assignment.

        Args:
            data: The scale data containing all measurements.

        Returns:
            Dictionary with only raw measurements (weight, impedance).
        """
        raw_measurements = {}
        if "weight" in data.measurements:
            raw_measurements["weight"] = data.measurements["weight"]
        if "impedance" in data.measurements:
            raw_measurements["impedance"] = data.measurements["impedance"]
        return raw_measurements

    def _validate_measurement(
        self, weight_kg: float | None, impedance: float | None
    ) -> bool:
        """Validate measurement types (defensive check for corrupted BLE data).

        Args:
            weight_kg: Weight in kilograms (can be None).
            impedance: Impedance in ohms (can be None).

        Returns:
            True if measurements are valid types, False otherwise.
        """
        # Type validation only - scale hardware determines valid ranges
        if weight_kg is not None and not isinstance(weight_kg, (int, float)):
            _LOGGER.warning(
                "Invalid weight type: expected int or float, got %s (value: %s)",
                type(weight_kg).__name__,
                weight_kg,
            )
            return False

        if impedance is not None and not isinstance(impedance, (int, float)):
            _LOGGER.warning(
                "Invalid impedance type: expected int or float, got %s (value: %s)",
                type(impedance).__name__,
                impedance,
            )
            return False

        return True

    def _cleanup_old_pending_measurements(self) -> None:
        """Clean up oldest pending measurements when limit is exceeded (FIFO)."""
        if len(self._pending_measurements) > self.MAX_PENDING_MEASUREMENTS:
            oldest_timestamp = next(iter(self._pending_measurements))
            pending_data = self._pending_measurements[oldest_timestamp]

            # Dismiss all mobile notifications for this measurement
            notified_services = pending_data.get("notified_mobile_services", [])
            for user_id, service_name in notified_services:
                tag = f"scale_measurement_{oldest_timestamp}"

                async def _safe_clear_notification(
                    service: str, notification_tag: str, notification_user_id: str
                ) -> None:
                    """Safely clear notification with error handling."""
                    try:
                        await self._hass.services.async_call(
                            "notify",
                            service,
                            {
                                "message": "clear_notification",
                                "data": {"tag": notification_tag},
                            },
                        )
                        _LOGGER.debug(
                            "Dismissed mobile notification for user %s on %s (tag: %s)",
                            notification_user_id,
                            service,
                            notification_tag,
                        )
                    except Exception as ex:
                        _LOGGER.error(
                            "Failed to clear notification (service: %s, tag: %s, error: %s)",
                            service,
                            notification_tag,
                            ex,
                        )

                self._hass.async_create_task(
                    _safe_clear_notification(service_name, tag, user_id)
                )

            del self._pending_measurements[oldest_timestamp]

            # Clean up the persistent notification for the oldest measurement
            self._ambiguous_notifications.discard(oldest_timestamp)
            persistent_notification.dismiss(
                self._hass, f"etekcity_scale_{self.address}_{oldest_timestamp}"
            )

            _LOGGER.debug(
                "Removed oldest pending measurement: %s (FIFO cleanup, max=%d)",
                oldest_timestamp,
                self.MAX_PENDING_MEASUREMENTS,
            )

            # Note: Don't call _notify_diagnostic_sensors() here as this method is always
            # called right before adding a new pending measurement (which will trigger notification)

    def _notify_diagnostic_sensors(self) -> None:
        """Notify diagnostic sensors about state changes (e.g., pending measurements updated).

        This is used to trigger updates to diagnostic sensors that display coordinator
        state (like pending measurements) rather than scale measurement data.

        Diagnostic sensors pull their data directly from coordinator state, so they
        only need a notification to refresh, not actual ScaleData.
        """
        # Notify diagnostic listeners (no data passed, they pull from coordinator)
        for listener_callback in self._diagnostic_listeners:
            try:
                listener_callback()
            except Exception as ex:
                _LOGGER.error("Error notifying diagnostic listener: %s", ex)

    @callback
    def update_listeners(self, data: ScaleData) -> None:
        """Update all registered listeners with multi-user routing.

        Args:
            data: The scale data to send to listeners.
        """
        if not data:
            _LOGGER.warning(
                "Received empty data update from scale (address: %s)", self.address
            )
            return

        # Log received measurements
        measurements = list(data.measurements.keys())
        _LOGGER.debug(
            "MEASUREMENT RECEIVED from scale %s with %d measurements: %s",
            self.address,
            len(measurements),
            ", ".join(measurements),
        )

        # Extract weight for person detection
        weight_kg = data.measurements.get("weight")
        if weight_kg is None:
            _LOGGER.warning(
                "No weight measurement in scale data (address: %s), cannot route to user",
                self.address,
            )
            return

        # Validate measurement ranges
        impedance = data.measurements.get("impedance")
        if not self._validate_measurement(weight_kg, impedance):
            _LOGGER.error(
                "Invalid measurement values, rejecting data (weight: %s kg, impedance: %s Ω)",
                weight_kg,
                impedance,
            )
            return

        # Create timestamp ONCE when measurement is received
        # This ensures consistent timestamps across all code paths (auto-assign, detection, pending)
        measurement_timestamp = datetime.now().isoformat()

        # Smart detection logic: Single user auto-assign (skip detection)
        if len(self._user_profiles) == 1:
            user_id = self._user_profiles[0].get(CONF_USER_ID)
            _LOGGER.debug(
                "Single user detected, auto-assigning measurement to user %s (weight: %.2f kg)",
                user_id,
                weight_kg,
            )
            self._route_to_user(user_id, data, timestamp=measurement_timestamp)
            _LOGGER.debug(
                "Finished processing measurement update (single user auto-assign)"
            )
            return

        # Run person detection (returns list of candidates: weight matches + users without history)
        # Location filtering is already applied by the detector
        candidates = self._person_detector.detect_person(weight_kg, self._user_profiles)

        # Fallback: If no candidates found, include all users
        # This prevents data loss when weight is out of tolerance for all users
        if not candidates:
            _LOGGER.debug(
                "No candidates detected for weight %.2f kg, falling back to all users",
                weight_kg,
            )
            candidates = [
                u.get(CONF_USER_ID)
                for u in self._user_profiles
                if u.get(CONF_USER_ID) is not None
            ]

        # Handle detection results
        if len(candidates) == 1:
            # Exactly one candidate - auto-assign
            auto_assign_user_id = candidates[0]
            _LOGGER.debug(
                "Single candidate (user %s) - auto-assigning measurement (weight: %.2f kg)",
                auto_assign_user_id,
                weight_kg,
            )
            self._route_to_user(
                auto_assign_user_id, data, timestamp=measurement_timestamp
            )
        elif len(candidates) > 1:
            # Multiple candidates - store as pending and notify
            # Reuse measurement_timestamp for consistency
            timestamp = measurement_timestamp
            # Store only raw measurements (body metrics will be calculated on assignment)
            raw_measurements = self._extract_raw_measurements(data)
            self._pending_measurements[timestamp] = {
                "measurements": raw_measurements,
                "candidates": candidates,
                "notified_mobile_services": [],  # Will be populated when notifications sent
            }

            # Keep only last N pending measurements (FIFO cleanup)
            self._cleanup_old_pending_measurements()

            # Schedule async notification (runs in background)
            async def _safe_create_notification() -> None:
                """Safely create ambiguous notification with error handling."""
                try:
                    await self._create_ambiguous_notification(
                        weight_kg, impedance, candidates, timestamp
                    )
                except Exception as ex:
                    _LOGGER.error(
                        "Failed to create ambiguous notification (timestamp: %s, error: %s)",
                        timestamp,
                        ex,
                    )

            self._hass.async_create_task(_safe_create_notification())

            # Notify diagnostic sensors about pending measurements update
            self._notify_diagnostic_sensors()

        _LOGGER.debug("Finished processing measurement update")

    def _route_to_user_internal(
        self, user_id: str, data: ScaleData, timestamp: str
    ) -> None:
        """Internal method to route measurement to a specific user's sensors without persisting.

        Args:
            user_id: The user ID to route to.
            data: The scale data to send.
            timestamp: ISO timestamp of when the measurement was received.
        """
        # Find user profile using O(1) dictionary lookup
        user_profile = self._user_profiles_by_id.get(user_id)
        if not user_profile:
            _LOGGER.error(
                "User profile not found for user_id: %s (cannot route measurement)",
                user_id,
            )
            return

        # Store measurement in user's weight history
        weight_kg = data.measurements.get("weight")
        impedance = data.measurements.get("impedance")

        if weight_kg is not None:
            # Add to persistent history
            self._add_measurement_to_history(user_id, timestamp, weight_kg, impedance)

        # Check if the added measurement is the newest (by timestamp)
        # If not, this is a backfill scenario - update sensors with current newest to refresh attributes
        newest_measurement = self.get_last_measurement(user_id)
        is_backfill = (
            newest_measurement and newest_measurement["timestamp"] != timestamp
        )

        if is_backfill:
            _LOGGER.debug(
                "Added historical measurement %s for user %s, will update sensors with current newest %s to refresh attributes",
                timestamp,
                user_id,
                newest_measurement["timestamp"],
            )
            # Build ScaleData from the current newest measurement (not the backfilled one)
            data = self._build_measurement_data_from_history(
                user_id, newest_measurement
            )
            # Route to sensors to refresh state and attributes
            for update_callback in self._user_callbacks.get(user_id, []):
                try:
                    update_callback(data)
                except Exception as ex:
                    _LOGGER.error(
                        "Error updating listener for user_id: %s (%s: %s)",
                        user_id,
                        type(ex).__name__,
                        ex,
                    )
            return

        # Calculate body metrics if enabled for this user (newest measurement scenario)
        if user_profile.get("body_metrics_enabled", False):
            try:
                from etekcity_esf551_ble.body_metrics import (
                    BodyMetrics,
                    Sex,
                    _as_dictionary,
                    _calc_age,
                )
                from datetime import date as dt_date

                weight_kg = data.measurements.get("weight")
                impedance = data.measurements.get("impedance")

                if weight_kg:
                    height_cm = user_profile.get("height")
                    user_name = user_profile.get("name", user_id)

                    if height_cm is None:
                        _LOGGER.warning(
                            "Missing height for user_id: %s, skipping body metrics calculation",
                            user_id,
                        )
                        height_m = None
                    elif not isinstance(height_cm, (int, float)) or height_cm <= 0:
                        _LOGGER.error(
                            "Invalid height for user_id: %s (height: %s cm, must be positive number)",
                            user_id,
                            height_cm,
                        )
                        height_m = None
                    else:
                        height_m = height_cm / 100.0

                    if height_m is not None:
                        if impedance:
                            birthdate_str = user_profile.get("birthdate")
                            if isinstance(birthdate_str, str):
                                birthdate = dt_date.fromisoformat(birthdate_str)
                            else:
                                birthdate = birthdate_str

                            sex_str = user_profile.get("sex", "Male")
                            sex = Sex.Male if sex_str == "Male" else Sex.Female
                            age = _calc_age(birthdate)
                            body_metrics = BodyMetrics(
                                weight_kg, height_m, age, sex, impedance
                            )
                            metrics_dict = _as_dictionary(body_metrics)

                            # Add body metrics to measurements
                            data.measurements.update(metrics_dict)
                            _LOGGER.debug(
                                "Added body metrics for user %s: %s",
                                user_name,
                                list(metrics_dict.keys()),
                            )
                        else:
                            _LOGGER.warning(
                                "No impedance measurement available for user %s, skipping impedance-dependent body metrics calculation",
                                user_id,
                            )
                            # Not going through the body metrics calculation, so we calculate BMI manually for now.
                            data.measurements["body_mass_index"] = (
                                floor(weight_kg / (height_m**2) * 100) / 100
                            )
            except (ValueError, TypeError, AttributeError) as ex:
                # Catch expected errors from invalid data
                _LOGGER.error(
                    "Error calculating body metrics for user_id: %s (%s: %s)",
                    user_id,
                    type(ex).__name__,
                    ex,
                )
            except Exception:
                # Catch unexpected errors and log with full traceback
                _LOGGER.exception(
                    "Unexpected error calculating body metrics for user_id: %s", user_id
                )

        # Route to user-specific listeners using direct callback registry
        for update_callback in self._user_callbacks.get(user_id, []):
            try:
                update_callback(data)
            except Exception as ex:
                _LOGGER.error(
                    "Error updating listener for user_id: %s (%s: %s)",
                    user_id,
                    type(ex).__name__,
                    ex,
                )

    def _route_to_user(self, user_id: str, data: ScaleData, timestamp: str) -> None:
        """Route measurement to a specific user's sensors.

        Args:
            user_id: The user ID to route to.
            data: The scale data to send.
            timestamp: ISO timestamp of when the measurement was received.
        """
        self._route_to_user_internal(user_id, data, timestamp)
        self._update_config_entry()

    async def _send_mobile_notifications_for_ambiguous_measurement(
        self,
        timestamp: str,
        weight_kg: float,
        impedance_ohms: float | None,
        candidates: list[str],
    ) -> list[tuple[str, str]]:
        """Send mobile notifications to candidate users.

        Groups candidates by mobile device and sends smart notifications:
        - Single user per device: "Is this yours?" with "Assign to Me" button
        - Multiple users per device: "Who stepped on?" with "Assign to Alice", "Assign to Bob" buttons

        Args:
            timestamp: ISO timestamp of the measurement
            weight_kg: Weight in kilograms
            impedance_ohms: Impedance in ohms (or None)
            candidates: List of candidate user_ids

        Returns:
            List of (user_id, service_name) tuples for services that were notified
        """
        notified_services = []

        # Format weight for display using coordinator's display unit
        if self._display_unit == WeightUnit.LB:
            weight_value = MassConverter.convert(
                weight_kg, UnitOfMass.KILOGRAMS, UnitOfMass.POUNDS
            )
            weight_display = f"{weight_value:.1f} lb"
        else:
            # Default to kg if display_unit is None or WeightUnit.KG
            weight_display = f"{weight_kg:.1f} kg"

        # Format time
        dt = datetime.fromisoformat(timestamp)
        time_display = dt.strftime("%I:%M %p").lstrip("0")

        # Notification tag (unique per measurement)
        tag = f"scale_measurement_{timestamp}"

        # Group candidates by mobile device service
        # Structure: {service_name: [(user_id, user_name), ...]}
        device_to_users: dict[str, list[tuple[str, str]]] = {}

        for user_id in candidates:
            user_profile = self._user_profiles_by_id.get(user_id)
            if not user_profile:
                continue

            user_name = user_profile.get(CONF_USER_NAME, "Unknown")
            mobile_services = user_profile.get(CONF_MOBILE_NOTIFY_SERVICES, [])

            if not mobile_services:
                _LOGGER.debug(
                    "No mobile notify services configured for user %s, "
                    "skipping mobile notification",
                    user_name,
                )
                continue

            # Add this user to each of their configured devices
            for service_name in mobile_services:
                if service_name not in device_to_users:
                    device_to_users[service_name] = []
                device_to_users[service_name].append((user_id, user_name))

        # Encode timestamp for safe embedding in action identifiers
        encoded_timestamp = quote(timestamp, safe="")
        # Use placeholder for empty string user_id (v1 legacy) to avoid encoding issues
        LEGACY_USER_ID_PLACEHOLDER = "__legacy__"

        # Send one notification per device with appropriate message and actions
        for service_name, users in device_to_users.items():
            try:
                # Determine if this is a single-user or multi-user device
                if len(users) == 1:
                    # Check if this device is associated with other users (not candidates)
                    # If so, we need to include the user's name to avoid ambiguity
                    user_id, user_name = users[0]
                    other_users_on_device = []
                    for profile in self._user_profiles:
                        profile_user_id = profile.get(CONF_USER_ID)
                        if profile_user_id is None:
                            continue
                        # Skip if this is the candidate user
                        if profile_user_id == user_id:
                            continue
                        # Check if this profile has this device configured
                        profile_mobile_services = profile.get(
                            CONF_MOBILE_NOTIFY_SERVICES, []
                        )
                        if service_name in profile_mobile_services:
                            other_users_on_device.append(
                                profile.get(CONF_USER_NAME, "Unknown")
                            )

                    # If device is shared with other users, include name in message/button
                    if other_users_on_device:
                        # Device is shared - make it clear which user this is for
                        message = f"{weight_display} at {time_display}. Is this {user_name}'s?"
                        button_title = f"Assign to {user_name}"
                        not_me_title = f"Not {user_name}"
                    else:
                        # Device is only for this user - can use generic "Me"
                        message = f"{weight_display} at {time_display}. Is this yours?"
                        button_title = "Assign to Me"
                        not_me_title = "Not Me"

                    # Use placeholder for empty string user_id (v1 legacy compatibility)
                    encoded_user_id = (
                        LEGACY_USER_ID_PLACEHOLDER
                        if user_id == ""
                        else quote(user_id, safe="")
                    )

                    actions = [
                        {
                            "action": f"SCALE_ASSIGN_{encoded_user_id}_{encoded_timestamp}",
                            "title": button_title,
                        },
                        {
                            "action": f"SCALE_NOT_ME_{encoded_user_id}_{encoded_timestamp}",
                            "title": not_me_title,
                        },
                    ]

                    action_data = {
                        "timestamp": timestamp,
                        "user_id": user_id,
                    }

                    _LOGGER.debug(
                        "Sending personalized notification to %s via %s%s",
                        user_name,
                        service_name,
                        f" (shared device with {', '.join(other_users_on_device)})"
                        if other_users_on_device
                        else "",
                    )
                else:
                    # Multi-user shared device notification
                    user_names = [name for _, name in users]
                    message = f"{weight_display} at {time_display}. Who stepped on?"

                    # Build action buttons (limit to first 3 users due to platform constraints)
                    actions = []
                    for user_id, user_name in users[:3]:
                        # Use placeholder for empty string user_id (v1 legacy compatibility)
                        encoded_user_id = (
                            LEGACY_USER_ID_PLACEHOLDER
                            if user_id == ""
                            else quote(user_id, safe="")
                        )
                        actions.append(
                            {
                                "action": f"SCALE_ASSIGN_{encoded_user_id}_{encoded_timestamp}",
                                "title": f"Assign to {user_name}",
                            }
                        )

                    # If more than 3 users, mention in message
                    if len(users) > 3:
                        remaining = len(users) - 3
                        overflow_names = ", ".join(user_names[3:])
                        message += f" (Tap for {', '.join(user_names[:3])}, +{remaining} more: {overflow_names})"

                    # Include all user_ids in action_data for fallback
                    action_data = {
                        "timestamp": timestamp,
                        "user_ids": [uid for uid, _ in users],
                    }

                    _LOGGER.debug(
                        "Sending multi-user notification to %s with %d candidates: %s",
                        service_name,
                        len(users),
                        ", ".join(user_names),
                    )

                await self._hass.services.async_call(
                    "notify",
                    service_name,
                    {
                        "title": "❓ Unassigned Scale Measurement",
                        "message": message,
                        "data": {
                            "tag": tag,
                            "group": "scale-measurements",
                            "channel": "Scale Measurements",
                            "importance": "default",
                            "actions": actions,
                            "action_data": action_data,
                        },
                    },
                )

                # Track all users notified via this service
                for user_id, user_name in users:
                    notified_services.append((user_id, service_name))

            except Exception as ex:
                _LOGGER.error(
                    "Failed to send mobile notification to %s: %s",
                    service_name,
                    ex,
                )

        return notified_services

    async def _create_ambiguous_notification(
        self,
        weight_kg: float,
        impedance: float | None,
        ambiguous_user_ids: list[str],
        timestamp: str,
    ) -> None:
        """Create an enhanced persistent notification for ambiguous measurements.

        Filters and ranks users intelligently:
        1. First shows users matching within tolerance (sorted by closeness)
        2. Then shows all other potential candidates.

        Args:
            weight_kg: The measured weight in kg.
            ambiguous_user_ids: list of user IDs that could match.
            timestamp: Timestamp of the measurement.
            impedance: Optional impedance measurement in ohms.
        """
        # Send mobile notifications first (async operation)
        # Resolve device info for notification context
        device_reg = dr.async_get(self._hass)
        device_entry = device_reg.async_get_device(
            connections={(CONNECTION_BLUETOOTH, self.address)}
        )
        device_id = device_entry.id if device_entry else "DEVICE_ID"
        device_name = device_entry.name if device_entry else self._device_name

        # Helper to convert weight to configured unit for display
        def _format_weight(value_kg: float, precision: int = 2) -> str:
            if self._display_unit == WeightUnit.LB:
                value = MassConverter.convert(
                    value_kg, UnitOfMass.KILOGRAMS, UnitOfMass.POUNDS
                )
                unit = "lb"
            else:
                value = value_kg
                unit = "kg"
            fmt = f"{value:.{precision}f} {unit}"
            return fmt

        # Categorize candidates for notification display:
        # - Matching users: Have usable history (show weight difference)
        # - Other users: No usable history (new users or stale data 90+ days)
        #
        # Candidates come from PersonDetector which includes:
        # 1. Users matching within adaptive tolerance
        # 2. Users without usable history (new users or stale history)
        #
        # If PersonDetector returns empty, coordinator fallback adds all users.
        matching_users = []  # (user_id, weight_diff, user_name) - users with history
        other_users = []  # (user_id, user_name) - new users without history

        # Get users with valid (usable) history using same logic as PersonDetector
        users_with_valid_history = self._person_detector.get_users_with_history(
            self._user_profiles
        )

        for user_id in ambiguous_user_ids:
            user_profile = self._user_profiles_by_id.get(user_id)
            if not user_profile:
                # Should never happen - would mean user was deleted between update_listeners() and here
                _LOGGER.warning(
                    "User profile %s not found in notification creation (should not happen)",
                    user_id,
                )
                continue

            user_name = user_profile.get(CONF_USER_NAME, user_id)

            # Use consistent definition of "usable history" from PersonDetector
            if user_id not in users_with_valid_history:
                # No usable history (new user OR stale history 90+ days)
                other_users.append((user_id, user_name))
                continue

            # User has usable history - get last measurement for ranking
            weight_history = user_profile.get(CONF_WEIGHT_HISTORY, [])
            last_weight = weight_history[-1]["weight_kg"]
            weight_diff = abs(weight_kg - last_weight)
            matching_users.append((user_id, weight_diff, user_name))

        # Sort matching users by weight difference (closest first)
        matching_users.sort(key=lambda x: x[1])

        # Sort other users alphabetically by name
        other_users.sort(key=lambda x: x[1])

        total_candidates = len(matching_users) + len(other_users)

        # Defensive check: This function should only be called for ambiguous (multiple) candidates
        # Single candidates are auto-assigned in update_listeners() before notification is created
        if total_candidates == 1:
            _LOGGER.warning(
                "Notification called with single candidate - coordinator should have auto-assigned."
            )

        # Continue with notification creation for multiple candidates
        elif total_candidates == 0:
            # Edge case: shouldn't happen but log if it does
            _LOGGER.error("Notification called with zero candidates.")
            return

        # Build the user list for the notification message
        user_list_items = []
        if matching_users:
            user_list_items.append("**Candidates:**")
            for user_id, diff, user_name in matching_users:
                user_id_display = '""' if user_id == "" else user_id
                user_list_items.append(
                    f"- **{user_name}** ({user_id_display}) — ±{_format_weight(diff, 1)}"
                )

        if other_users:
            if not user_list_items:
                user_list_items.append("**Candidates:**")

            for user_id, user_name in other_users:
                user_id_display = '""' if user_id == "" else user_id
                user_list_items.append(f"- **{user_name}** ({user_id_display})")

        user_list_str = "\n".join(user_list_items)

        # Build measurement info
        measurement_info = f"Weight: **{_format_weight(weight_kg)}**"
        if impedance is not None:
            measurement_info += f"  \nImpedance: **{impedance:.0f} Ω**"

        message = (
            f"**Scale: {device_name}**\n\n"
            f"**Multiple users could match this measurement**\n\n"
            f"{measurement_info}\n"
            f"Timestamp: `{timestamp}`\n\n"
            f"{user_list_str}\n\n"
            "**To assign this measurement:**\n"
            "1. Copy the service call below\n"
            "2. Go to **Developer Tools → Actions**\n"
            "3. Paste and select the correct `user_id`\n"
            "4. Click **Perform Action**\n\n"
            f"```yaml\n"
            f"action: etekcity_fitness_scale_ble.assign_measurement\n"
            f"data:\n"
            f"  device_id: {device_id}\n"
            f'  timestamp: "{timestamp}"\n'
            f'  user_id: "<SELECT_USER_ID_FROM_ABOVE>"\n'
            f"```\n\n"
            "This notification will auto-dismiss once the measurement is assigned."
        )

        # Send mobile app notifications to relevant users (only if multiple candidates)
        notified_services = (
            await self._send_mobile_notifications_for_ambiguous_measurement(
                timestamp, weight_kg, impedance, ambiguous_user_ids
            )
        )

        # Store notified services in pending measurement for later dismissal
        if timestamp in self._pending_measurements:
            self._pending_measurements[timestamp]["notified_mobile_services"] = (
                notified_services
            )

        # Track this ambiguous notification
        self._ambiguous_notifications.add(timestamp)

        notification_id = f"etekcity_scale_{self.address}_{timestamp}"
        _LOGGER.debug(
            "Creating persistent notification with ID: %s",
            notification_id,
        )
        persistent_notification.create(
            self._hass,
            message,
            title=f"{device_name}: Choose User",
            notification_id=notification_id,
        )

        # Update diagnostic sensors to reflect new pending measurement
        self._notify_diagnostic_sensors()

    def get_user_profiles(self) -> list[dict]:
        """Get all user profiles.

        Returns:
            list of user profile dictionaries.
        """
        return self._user_profiles

    def _build_measurement_data_from_history(
        self, user_id: str, measurement: dict
    ) -> ScaleData:
        """Build a complete ScaleData with body metrics from a historical measurement.

        Takes a measurement from history (which only has weight_kg and impedance_ohm)
        and creates a full ScaleData object with recalculated body metrics based on
        the user's current profile.

        Args:
            user_id: The user ID
            measurement: Measurement dict from history with 'weight_kg' and optionally 'impedance_ohm'

        Returns:
            ScaleData object with weight, impedance, and recalculated body metrics
        """
        user_profile = self._user_profiles_by_id.get(user_id)
        if not user_profile:
            _LOGGER.error(
                "User profile not found for user_id: %s (cannot build measurement data)",
                user_id,
            )
            return ScaleData(measurements={})

        # Convert history format to measurement format
        measurements = {"weight": measurement["weight_kg"]}
        if "impedance_ohm" in measurement:
            measurements["impedance"] = measurement["impedance_ohm"]

        # Calculate body metrics if enabled for this user
        if user_profile.get("body_metrics_enabled", False):
            try:
                from etekcity_esf551_ble.body_metrics import (
                    BodyMetrics,
                    Sex,
                    _as_dictionary,
                    _calc_age,
                )
                from datetime import date as dt_date

                weight_kg = measurements.get("weight")
                impedance = measurements.get("impedance")

                if weight_kg:
                    height_cm = user_profile.get("height")
                    user_name = user_profile.get("name", user_id)

                    if height_cm is None:
                        _LOGGER.warning(
                            "Missing height for user_id: %s, skipping body metrics calculation",
                            user_id,
                        )
                        height_m = None
                    elif not isinstance(height_cm, (int, float)) or height_cm <= 0:
                        _LOGGER.error(
                            "Invalid height for user_id: %s (height: %s cm, must be positive number), skipping body metrics",
                            user_id,
                            height_cm,
                        )
                        height_m = None
                    else:
                        height_m = height_cm / 100.0

                    if height_m is not None:
                        if impedance:
                            birthdate_str = user_profile.get("birthdate")
                            if isinstance(birthdate_str, str):
                                birthdate = dt_date.fromisoformat(birthdate_str)
                            else:
                                birthdate = birthdate_str

                            sex_str = user_profile.get("sex", "Male")
                            sex = Sex.Male if sex_str == "Male" else Sex.Female
                            age = _calc_age(birthdate)
                            body_metrics = BodyMetrics(
                                weight_kg, height_m, age, sex, impedance
                            )
                            metrics_dict = _as_dictionary(body_metrics)

                            # Add body metrics to measurements
                            measurements.update(metrics_dict)
                            _LOGGER.debug(
                                "Recalculated body metrics for user %s from history: %s",
                                user_name,
                                list(metrics_dict.keys()),
                            )
                        else:
                            # No impedance - calculate BMI only
                            measurements["body_mass_index"] = (
                                floor(weight_kg / (height_m**2) * 100) / 100
                            )
            except (ValueError, TypeError, AttributeError) as ex:
                # Catch expected errors from invalid data
                _LOGGER.error(
                    "Error recalculating body metrics for user_id: %s (%s: %s)",
                    user_id,
                    type(ex).__name__,
                    ex,
                )
            except Exception:
                # Catch unexpected errors and log with full traceback
                _LOGGER.exception(
                    "Unexpected error recalculating body metrics for user_id: %s",
                    user_id,
                )

        return ScaleData(measurements=measurements)

    def get_pending_measurements(self) -> dict[str, dict]:
        """Get all pending measurements.

        Returns:
            Dictionary mapping timestamp to dict with keys:
            - "measurements": raw_measurements_dict (weight, impedance)
            - "candidates": list of candidate user_ids
            - "notified_mobile_services": list of (user_id, service_name) tuples
        """
        return self._pending_measurements

    def assign_pending_measurement(self, timestamp: str, user_id: str) -> bool:
        """Manually assign a pending measurement to a user.

        Pending measurements contain only raw scale data (weight, impedance).
        Body metrics are calculated fresh based on the assigned user's profile.

        Args:
            timestamp: ISO timestamp of the pending measurement.
            user_id: The user ID to assign the measurement to.

        Returns:
            True if assignment succeeded, False otherwise.
        """
        # Validate user_id exists
        if user_id not in self._user_profiles_by_id:
            _LOGGER.error(
                "User profile not found for user_id: %s (cannot assign pending measurement)",
                user_id,
            )
            return False

        if timestamp not in self._pending_measurements:
            _LOGGER.warning(
                "No pending measurement found for timestamp: %s (cannot assign to user_id: %s)",
                timestamp,
                user_id,
            )
            return False

        pending_data = self._pending_measurements.pop(timestamp)
        measurements = pending_data["measurements"]
        notified_services = pending_data.get("notified_mobile_services", [])

        _LOGGER.debug(
            "Manually assigned measurement from %s to user %s (weight: %.2f kg)",
            timestamp,
            user_id,
            measurements.get("weight"),
        )

        # Create a ScaleData object with raw measurements and route to the user
        # Body metrics will be calculated by _route_to_user() based on the user's profile
        # Pass the original timestamp to preserve measurement time
        scale_data = ScaleData(measurements=measurements)
        self._route_to_user(user_id, scale_data, timestamp=timestamp)

        # Clean up tracking structures
        self._ambiguous_notifications.discard(timestamp)

        # Dismiss the persistent notification
        notification_id = f"etekcity_scale_{self.address}_{timestamp}"
        _LOGGER.debug(
            "Dismissing persistent notification with ID: %s",
            notification_id,
        )
        persistent_notification.dismiss(
            self._hass,
            notification_id=notification_id,
        )

        # Dismiss all mobile notifications for this measurement
        tag = f"scale_measurement_{timestamp}"
        for user_id_notified, service_name in notified_services:
            self._hass.async_create_task(
                self._hass.services.async_call(
                    "notify",
                    service_name,
                    {"message": "clear_notification", "data": {"tag": tag}},
                )
            )
            _LOGGER.debug(
                "Dismissed mobile notification for user %s on %s (tag: %s)",
                user_id_notified,
                service_name,
                tag,
            )

        # Notify diagnostic sensors about pending measurements update
        self._notify_diagnostic_sensors()

        return True

    def reassign_user_measurement(
        self, from_user_id: str, to_user_id: str, timestamp: str | None = None
    ) -> bool:
        """Reassign a measurement from one user to another.

        Args:
            from_user_id: The user ID to take the measurement from.
            to_user_id: The user ID to assign the measurement to.
            timestamp: Specific timestamp to reassign, or None for newest (backward compatible).

        Returns:
            True if reassignment succeeded, False otherwise.

        Note:
            Caller is responsible for validating that both user IDs exist.
            Service handlers in __init__.py perform this validation.
        """
        # Get measurement from source user's history
        history = self.get_user_history(from_user_id)
        if not history:
            _LOGGER.warning(
                "Cannot reassign from user %s: no measurement history found",
                from_user_id,
            )
            return False

        # Find the measurement to reassign
        measurement_to_reassign = None
        if timestamp is None:
            # Backward compatible: reassign newest
            measurement_to_reassign = history[-1]
            measurement_timestamp = measurement_to_reassign["timestamp"]
        else:
            # Find specific timestamp
            for m in history:
                if m["timestamp"] == timestamp:
                    measurement_to_reassign = m
                    measurement_timestamp = timestamp
                    break

        if not measurement_to_reassign:
            _LOGGER.warning(
                "Timestamp %s not found in history for user %s", timestamp, from_user_id
            )
            return False

        # Convert history format to measurements format
        # History uses "weight_kg" and "impedance_ohm", convert to "weight" and "impedance"
        measurements = {
            "weight": measurement_to_reassign["weight_kg"],
            "timestamp": measurement_timestamp,
        }
        if "impedance_ohm" in measurement_to_reassign:
            measurements["impedance"] = measurement_to_reassign["impedance_ohm"]

        _LOGGER.debug(
            "Retrieved measurement from history for user %s: weight=%.2f kg%s, timestamp=%s",
            from_user_id,
            measurements["weight"],
            f", impedance={measurements.get('impedance')} Ω"
            if "impedance" in measurements
            else "",
            measurement_timestamp,
        )

        # Validate target user exists
        if to_user_id not in self._user_profiles_by_id:
            _LOGGER.error(
                "User profile not found for user_id: %s (cannot reassign measurement)",
                to_user_id,
            )
            return False

        _LOGGER.debug(
            "Reassigning raw measurement from user %s to user %s (weight: %.2f kg%s, timestamp: %s)",
            from_user_id,
            to_user_id,
            measurements.get("weight"),
            f", impedance: {measurements.get('impedance')} Ω"
            if "impedance" in measurements
            else "",
            measurement_timestamp if measurement_timestamp else "not available",
        )

        # Create ScaleData with only raw measurements
        # Body metrics will be recalculated by _route_to_user_internal() based on target user's profile
        scale_data = ScaleData(measurements=measurements)

        # Remove measurement from source user (this updates source user's sensors)
        # Use internal method to avoid persisting - we'll persist once at the end
        if not self._remove_user_measurement_internal(
            from_user_id, measurement_timestamp
        ):
            return False

        # Route to target user (this will add to target's history)
        # Pass the original timestamp to preserve measurement time
        # Use internal method to avoid persisting - we'll persist once at the end
        self._route_to_user_internal(
            to_user_id, scale_data, timestamp=measurement_timestamp
        )

        # Persist changes once after both operations complete
        self._update_config_entry()

        _LOGGER.debug("=== REASSIGN COMPLETE ===")
        self._log_user_history(from_user_id, "source user (after removal)")
        self._log_user_history(to_user_id, "target user (after adding)")

        return True

    def _remove_user_measurement_internal(
        self, user_id: str, timestamp: str | None = None
    ) -> bool:
        """Internal method to remove a measurement from user's history without persisting.

        Args:
            user_id: The user ID to remove the measurement from.
            timestamp: Specific timestamp to remove, or None for newest (backward compatible).

        Returns:
            True if removal succeeded, False otherwise.
        """
        if timestamp:
            _LOGGER.debug(
                "Removing measurement with timestamp %s for user %s", timestamp, user_id
            )
        else:
            _LOGGER.debug("Removing newest measurement for user %s", user_id)

        # Remove from user's history
        user_profile = self._user_profiles_by_id.get(user_id)
        if not user_profile:
            _LOGGER.error(
                "User profile not found for user_id: %s (cannot remove measurement)",
                user_id,
            )
            return False

        history = user_profile.get(CONF_WEIGHT_HISTORY, [])
        if not history:
            _LOGGER.warning(
                "No measurements in history for user_id: %s (cannot remove measurement)",
                user_id,
            )
            return False

        # Find and remove the measurement
        removed = None
        if timestamp is None:
            # Backward compatible: remove newest (last in sorted list)
            removed = history.pop()
        else:
            # Remove specific timestamp
            for i, m in enumerate(history):
                if m["timestamp"] == timestamp:
                    removed = history.pop(i)
                    break

        if not removed:
            _LOGGER.warning(
                "Timestamp %s not found in history for user %s", timestamp, user_id
            )
            return False

        _LOGGER.debug(
            "Removed measurement from user %s history: weight=%.2f kg, timestamp=%s",
            user_id,
            removed.get("weight_kg"),
            removed.get("timestamp"),
        )
        self._log_user_history(user_id, "after removing measurement")

        # Update user's sensors with their new last measurement (after removal)
        # This recalculates body metrics from the remaining measurement
        last_measurement = self.get_last_measurement(user_id)
        if last_measurement:
            # User still has measurements - update sensors with recalculated body metrics
            update_data = self._build_measurement_data_from_history(
                user_id, last_measurement
            )
        else:
            # User has no more measurements - send empty data
            # Sensors will mark themselves unavailable when their key is missing
            update_data = ScaleData(measurements={})

        for update_callback in self._user_callbacks.get(user_id, []):
            try:
                update_callback(update_data)
            except Exception as ex:
                _LOGGER.error(
                    "Error updating sensor for user_id: %s (%s: %s)",
                    user_id,
                    type(ex).__name__,
                    ex,
                )

        return True

    def remove_user_measurement(
        self, user_id: str, timestamp: str | None = None
    ) -> bool:
        """Remove a measurement from user's history.

        Args:
            user_id: The user ID to remove the measurement from.
            timestamp: Specific timestamp to remove, or None for newest (backward compatible).

        Returns:
            True if removal succeeded, False otherwise.
        """
        if not self._remove_user_measurement_internal(user_id, timestamp):
            return False
        self._update_config_entry()
        return True
