"""Coordinator for the etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import platform
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

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
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import device_registry as dr  # NEW import
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH
from homeassistant.const import UnitOfMass
from homeassistant.util.unit_conversion import MassConverter

from .const import (
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_USER_ID,
    CONF_USER_NAME,
    DOMAIN,
    get_sensor_unique_id,
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
        # User-specific callback registry: user_id -> list of callbacks
        self._user_callbacks: dict[str, list[Callable[[ScaleData], None]]] = {}
        self._user_profiles = user_profiles
        # User profiles dictionary for O(1) lookup efficiency
        self._user_profiles_by_id: dict[str, dict] = {}
        v1_legacy_count = 0  # Track number of v1 legacy users (empty string user_id)

        for profile in user_profiles:
            user_id = profile.get(CONF_USER_ID)
            if user_id is not None:
                self._user_profiles_by_id[user_id] = profile
                # Count v1 legacy users (empty string reserved for v1 compatibility)
                if user_id == "":
                    v1_legacy_count += 1
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

        self._person_detector = PersonDetector(hass, device_name, DOMAIN)
        # Pending measurements awaiting manual assignment
        # Structure: {timestamp: dict with keys:
        #   - "measurements": raw_measurements_dict (weight, impedance)
        #   - "candidates": list of candidate user_ids
        #   - "notified_mobile_services": list of (user_id, service_name) tuples
        # }
        self._pending_measurements: dict[str, dict] = {}
        # Storage for reassignment/removal features
        self._last_user_measurement: dict[
            str, dict
        ] = {}  # user_id -> raw measurements dict
        self._ambiguous_notifications: set[str] = (
            set()
        )  # active notification timestamps

    def set_display_unit(self, unit: WeightUnit) -> None:
        """Set the display unit for the scale.

        Args:
            unit: The weight unit to display on the scale.
        """
        _LOGGER.debug("Setting display unit to: %s", unit.name)
        self._display_unit = unit
        if self._client:
            self._client.display_unit = unit

    async def _get_bluetooth_scanner(self) -> BaseBleakScanner | None:
        """Get the optimal Bluetooth scanner based on available resources.

        Returns:
            A configured Bluetooth scanner or None if no scanner could be created.
        """
        try:
            manager = self._hass.data.get("bluetooth_manager")
            if not manager:
                _LOGGER.warning("Bluetooth manager not available")
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
                            None, None, "active", esphome_clients
                        )
                        _LOGGER.debug(
                            "Created hybrid scanner with native and proxy support"
                        )
                    else:
                        scanner = BleakScannerESPHome(
                            None, None, "active", esphome_clients
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

            return scanner
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
            scanner = await self._get_bluetooth_scanner()

            # Initialize client (always use basic client, body metrics calculated per-user)
            try:
                _LOGGER.debug("Initializing new EtekcitySmartFitnessScale client")
                self._client = EtekcitySmartFitnessScale(
                    self.address,
                    self.update_listeners,
                    self._display_unit,
                    bleak_scanner_backend=scanner,
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
                _LOGGER.error("Failed to connect to scale at %s: %s", self.address, err)
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

    def _registration_changed(self, _: HaScannerRegistration) -> None:
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
                _LOGGER.error("Failed to start ScaleDataUpdateCoordinator: %s", ex)
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

        _LOGGER.debug("ScaleDataUpdateCoordinator stopped successfully")

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
            _LOGGER.warning("Invalid weight type: %s", type(weight_kg))
            return False

        if impedance is not None and not isinstance(impedance, (int, float)):
            _LOGGER.warning("Invalid impedance type: %s", type(impedance))
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
                self._hass.async_create_task(
                    self._hass.services.async_call(
                        "notify",
                        service_name,
                        {"message": "clear_notification", "data": {"tag": tag}},
                    )
                )
                _LOGGER.debug(
                    "Dismissed mobile notification for user %s on %s (tag: %s)",
                    user_id,
                    service_name,
                    tag,
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
        """
        # Create empty scale data to trigger sensor state updates
        empty_data = ScaleData(measurements={})

        # Notify all general listeners (includes diagnostic sensors)
        for listener_callback in self._listeners.values():
            try:
                listener_callback(empty_data)
            except Exception as ex:
                _LOGGER.error("Error notifying listener: %s", ex)

    @callback
    def update_listeners(self, data: ScaleData) -> None:
        """Update all registered listeners with multi-user routing.

        Args:
            data: The scale data to send to listeners.
        """
        if not data:
            _LOGGER.warning("Received empty data update from scale %s", self.address)
            return

        # Log received measurements
        measurements = list(data.measurements.keys())
        _LOGGER.debug(
            "Received data update from scale %s with %d measurements: %s",
            self.address,
            len(measurements),
            ", ".join(measurements),
        )

        # Extract weight for person detection
        weight_kg = data.measurements.get("weight")
        if weight_kg is None:
            _LOGGER.warning("No weight measurement in scale data, cannot route to user")
            return

        # Validate measurement ranges
        impedance = data.measurements.get("impedance")
        if not self._validate_measurement(weight_kg, impedance):
            _LOGGER.error(
                "Invalid measurement values, rejecting data (weight: %s, impedance: %s)",
                weight_kg,
                impedance,
            )
            return

        # Smart detection logic: Single user auto-assign (skip detection)
        if len(self._user_profiles) == 1:
            user_id = self._user_profiles[0].get(CONF_USER_ID)
            _LOGGER.debug(
                "Single user detected, auto-assigning measurement to user %s (weight: %.2f kg)",
                user_id,
                weight_kg,
            )
            self._route_to_user(user_id, data)
            return

        # Run person detection (matches users within weight tolerance)
        detected_user_id, ambiguous_user_ids = self._person_detector.detect_person(
            weight_kg, self._user_profiles
        )

        # Check which users have measurement history
        users_with_history = self._person_detector.get_users_with_history(
            self._user_profiles
        )
        users_without_history = [
            u.get(CONF_USER_ID)
            for u in self._user_profiles
            if u.get(CONF_USER_ID) is not None
            and u.get(CONF_USER_ID) not in users_with_history
        ]

        # Compile list of all possible matches:
        # - Likely matches from detector (detected_user_id or ambiguous_user_ids)
        # - All users without history (always included)
        all_possible_matches = set()

        # Add likely matches from detector
        if detected_user_id:
            # Single match from detector
            all_possible_matches.add(detected_user_id)
        elif ambiguous_user_ids:
            # Multiple matches from detector
            all_possible_matches.update(ambiguous_user_ids)

        # Always include users without history in the list
        all_possible_matches.update(users_without_history)

        # If no possible matches were found (e.g., out of tolerance for all users with history)
        # then per the requirement, we make all users possible matches.
        if not all_possible_matches:
            all_possible_matches.update(
                u.get(CONF_USER_ID)
                for u in self._user_profiles
                if u.get(CONF_USER_ID) is not None
            )

        # Convert to list for consistent ordering and processing
        all_possible_matches = list(all_possible_matches)

        # Apply location-based filtering as the final step, but ensure we don't end up with zero
        filtered_matches = self._person_detector._filter_candidates_by_location(
            all_possible_matches, self._user_profiles
        )

        if filtered_matches:
            all_possible_matches = filtered_matches

        # Handle detection results
        if len(all_possible_matches) == 1:
            # Exactly one possible match - auto-assign
            auto_assign_user_id = all_possible_matches[0]
            _LOGGER.debug(
                "Single possible match (user %s) - auto-assigning measurement (weight: %.2f kg)",
                auto_assign_user_id,
                weight_kg,
            )
            self._route_to_user(auto_assign_user_id, data)
        elif len(all_possible_matches) > 1:
            # Multiple possible matches - store as pending and notify
            # The `all_possible_matches` list is now the definitive list of candidates.
            timestamp = datetime.now().isoformat()
            # Store only raw measurements (body metrics will be calculated on assignment)
            raw_measurements = self._extract_raw_measurements(data)
            self._pending_measurements[timestamp] = {
                "measurements": raw_measurements,
                "candidates": all_possible_matches,
                "notified_mobile_services": [],  # Will be populated when notifications sent
            }

            # Keep only last N pending measurements (FIFO cleanup)
            self._cleanup_old_pending_measurements()

            # Schedule async notification (runs in background)
            self._hass.async_create_task(
                self._create_ambiguous_notification(
                    weight_kg, impedance, all_possible_matches, timestamp
                )
            )

            # Notify diagnostic sensors about pending measurements update
            self._notify_diagnostic_sensors()

    def _route_to_user(self, user_id: str, data: ScaleData) -> None:
        """Route measurement to a specific user's sensors.

        Args:
            user_id: The user ID to route to.
            data: The scale data to send.
        """
        # Find user profile using O(1) dictionary lookup
        user_profile = self._user_profiles_by_id.get(user_id)
        if not user_profile:
            _LOGGER.error("User profile not found for user_id: %s", user_id)
            return

        # Store raw measurements for this user (for reassignment/removal)
        weight_kg = data.measurements.get("weight")
        impedance = data.measurements.get("impedance")
        timestamp = datetime.now().isoformat()

        if weight_kg is not None:
            self._last_user_measurement[user_id] = {
                "weight": weight_kg,
                "impedance": impedance,
                "timestamp": timestamp,
            }
            _LOGGER.debug(
                "Stored measurement for user %s: weight=%.2f kg",
                user_id,
                weight_kg,
            )

        # Calculate body metrics if enabled for this user
        if user_profile.get("body_metrics_enabled", False):
            try:
                from etekcity_esf551_ble.body_metrics import (
                    BodyMetrics,
                    Sex as ESFSex,
                    _as_dictionary,
                    _calc_age,
                )
                from datetime import date as dt_date

                weight_kg = data.measurements.get("weight")
                impedance = data.measurements.get("impedance")

                if weight_kg and impedance:
                    birthdate_str = user_profile.get("birthdate")
                    if isinstance(birthdate_str, str):
                        birthdate = dt_date.fromisoformat(birthdate_str)
                    else:
                        birthdate = birthdate_str

                    sex_str = user_profile.get("sex", "Male")
                    sex = ESFSex.Male if sex_str == "Male" else ESFSex.Female
                    height_m = user_profile.get("height") / 100.0
                    age = _calc_age(birthdate)
                    body_metrics = BodyMetrics(weight_kg, height_m, age, sex, impedance)
                    metrics_dict = _as_dictionary(body_metrics)

                    # Add body metrics to measurements
                    data.measurements.update(metrics_dict)
                    _LOGGER.debug(
                        "Added body metrics for user %s: %s",
                        user_profile.get("name", user_id),
                        list(metrics_dict.keys()),
                    )
            except Exception as ex:
                _LOGGER.error(
                    "Error calculating body metrics for user %s: %s", user_id, ex
                )

        # Route to user-specific listeners using direct callback registry
        for update_callback in self._user_callbacks.get(user_id, []):
            try:
                update_callback(data)
            except Exception as ex:
                _LOGGER.error("Error updating listener for user %s: %s", user_id, ex)

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

        # Send one notification per device with appropriate message and actions
        for service_name, users in device_to_users.items():
            try:
                # Determine if this is a single-user or multi-user device
                if len(users) == 1:
                    # Personalized notification for single user
                    user_id, user_name = users[0]
                    message = f"{weight_display} at {time_display}. Is this yours?"

                    actions = [
                        {
                            "action": f"SCALE_ASSIGN_{user_id}_{timestamp}",
                            "title": "Assign to Me",
                        },
                        {
                            "action": f"SCALE_NOT_ME_{user_id}_{timestamp}",
                            "title": "Not Me",
                        },
                    ]

                    action_data = {
                        "timestamp": timestamp,
                        "user_id": user_id,
                    }

                    _LOGGER.debug(
                        "Sending personalized notification to %s via %s",
                        user_name,
                        service_name,
                    )
                else:
                    # Multi-user shared device notification
                    user_names = [name for _, name in users]
                    message = f"{weight_display} at {time_display}. Who stepped on?"

                    # Build action buttons (limit to first 3 users due to platform constraints)
                    actions = []
                    for user_id, user_name in users[:3]:
                        actions.append(
                            {
                                "action": f"SCALE_ASSIGN_{user_id}_{timestamp}",
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
        from .person_detector import WEIGHT_TOLERANCE_KG

        # Send mobile notifications first (async operation)
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

        # Get entity registry for weight lookups
        entity_reg = er.async_get(self._hass)

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

        # Categorize users: those within tolerance vs all others
        matching_users = []  # (user_id, weight_diff, user_name)
        other_users = []  # (user_id, user_name)

        for user_id in ambiguous_user_ids:
            user_profile = self._user_profiles_by_id.get(user_id)
            if not user_profile:
                continue

            user_name = user_profile.get(CONF_USER_NAME, user_id)

            # Try to find user's current weight to calculate a difference
            weight_diff = None
            try:
                sensor_unique_id = get_sensor_unique_id(
                    self._device_name, user_id, "weight"
                )
                sensor_entity_id = entity_reg.async_get_entity_id(
                    "sensor", DOMAIN, sensor_unique_id
                )

                if sensor_entity_id:
                    sensor_state = self._hass.states.get(sensor_entity_id)
                    if sensor_state and sensor_state.state not in (
                        "unknown",
                        "unavailable",
                    ):
                        # Convert sensor state to kg (handles unit conversion if display unit is pounds)
                        last_weight_kg = self._person_detector.sensor_state_to_kg(
                            sensor_state
                        )
                        if last_weight_kg is not None:
                            weight_diff = abs(weight_kg - last_weight_kg)
                        else:
                            weight_diff = None
                    else:
                        weight_diff = None
                else:
                    weight_diff = None
            except (ValueError, TypeError, AttributeError):
                weight_diff = None  # Treat parse errors as no valid history

            if weight_diff is not None and weight_diff <= WEIGHT_TOLERANCE_KG:
                matching_users.append((user_id, weight_diff, user_name))
            else:
                other_users.append((user_id, user_name))

        # Sort matching users by weight difference (closest first)
        matching_users.sort(key=lambda x: x[1])

        # Sort other users alphabetically by name
        other_users.sort(key=lambda x: x[1])

        total_candidates = len(matching_users) + len(other_users)
        if total_candidates == 1:
            # Single candidate - auto-assign without notification
            if matching_users:
                auto_assign_user_id = matching_users[0][0]
                auto_assign_user_name = matching_users[0][2]
                diff = matching_users[0][1]
                _LOGGER.debug(
                    "Single candidate after filtering, auto-assigning to %s (weight diff: ±%.2f kg)",
                    auto_assign_user_name,
                    diff,
                )
            else:
                auto_assign_user_id = other_users[0][0]
                auto_assign_user_name = other_users[0][1]
                _LOGGER.debug(
                    "Single candidate with no history, auto-assigning to %s",
                    auto_assign_user_name,
                )

            # Get raw measurements before removing from pending
            raw_measurements = {}
            if timestamp in self._pending_measurements:
                pending_data = self._pending_measurements[timestamp]
                raw_measurements = pending_data["measurements"]
                # Remove from pending
                del self._pending_measurements[timestamp]
                self._ambiguous_notifications.discard(timestamp)

            if not raw_measurements:
                # Extract from current data if not in pending
                raw_measurements = {
                    "weight": weight_kg,
                }
                if impedance is not None:
                    raw_measurements["impedance"] = impedance

            # Create ScaleData and route to user
            scale_data = ScaleData(measurements=raw_measurements)
            self._route_to_user(auto_assign_user_id, scale_data)
            return

        # Build the user list for the notification message
        user_list_items = []
        if matching_users:
            user_list_items.append("**Candidates:**")
            for user_id, diff, user_name in matching_users:
                user_list_items.append(
                    f"- **{user_name}** (`{user_id}`) — ±{_format_weight(diff, 1)}"
                )

        if other_users:
            if not user_list_items:
                user_list_items.append("**Candidates:**")

            for user_id, user_name in other_users:
                user_list_items.append(f"- **{user_name}** (`{user_id}`)")

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

        # Track this ambiguous notification
        self._ambiguous_notifications.add(timestamp)

        persistent_notification.create(
            self._hass,
            message,
            title=f"{device_name}: Choose User",
            notification_id=f"etekcity_scale_{self.address}_{timestamp}",
        )

    def get_user_profiles(self) -> list[dict]:
        """Get all user profiles.

        Returns:
            list of user profile dictionaries.
        """
        return self._user_profiles

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
            _LOGGER.error("User %s not found in user profiles", user_id)
            return False

        if timestamp not in self._pending_measurements:
            _LOGGER.warning("No pending measurement found for timestamp: %s", timestamp)
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
        scale_data = ScaleData(measurements=measurements)
        self._route_to_user(user_id, scale_data)

        # Clean up tracking structures
        self._ambiguous_notifications.discard(timestamp)

        # Dismiss the persistent notification
        persistent_notification.dismiss(
            self._hass,
            notification_id=f"etekcity_scale_{self.address}_{timestamp}",
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

    def reassign_user_measurement(self, from_user_id: str, to_user_id: str) -> bool:
        """Reassign a user's last measurement to a different user.

        This works both for the current session (using in-memory data) and after
        a restart (by retrieving data from sensor state).

        Args:
            from_user_id: The user ID to take the measurement from.
            to_user_id: The user ID to assign the measurement to.

        Returns:
            True if reassignment succeeded, False otherwise.

        Note:
            Caller is responsible for validating that both user IDs exist.
            Service handlers in __init__.py perform this validation.
        """
        # Get measurement from source user (try memory first, then sensor state)
        measurements = self._last_user_measurement.get(from_user_id)

        if not measurements:
            # Try to get from sensor states using Entity Registry
            # We'll retrieve only RAW measurements (weight, impedance)
            # Body metrics are calculated and should be recalculated for target user
            entity_reg = er.async_get(self._hass)
            measurements = {}

            # Only retrieve raw scale measurements (not calculated body metrics)
            sensor_keys = ["weight", "impedance"]

            # Retrieve each sensor's state
            for sensor_key in sensor_keys:
                # Construct the unique_id for this sensor using helper function
                sensor_unique_id = get_sensor_unique_id(
                    self._device_name, from_user_id, sensor_key
                )

                # Look up the entity_id
                sensor_entity_id = entity_reg.async_get_entity_id(
                    "sensor", DOMAIN, sensor_unique_id
                )

                if sensor_entity_id:
                    sensor_state = self._hass.states.get(sensor_entity_id)
                    if sensor_state and sensor_state.state not in (
                        "unknown",
                        "unavailable",
                    ):
                        try:
                            if sensor_key == "weight":
                                # For weight, convert to kg (native unit) if needed
                                value = float(sensor_state.state)
                                unit = sensor_state.attributes.get(
                                    "unit_of_measurement"
                                )
                                if unit == UnitOfMass.POUNDS:
                                    # Convert pounds to kilograms (native unit)
                                    value = MassConverter.convert(
                                        value, UnitOfMass.POUNDS, UnitOfMass.KILOGRAMS
                                    )
                                measurements[sensor_key] = value
                            else:
                                # For impedance, use value as-is (no unit conversion)
                                value = float(sensor_state.state)
                                measurements[sensor_key] = value

                            _LOGGER.debug(
                                "Retrieved %s from sensor state for user %s: %s",
                                sensor_key,
                                from_user_id,
                                measurements[sensor_key],
                            )
                        except (ValueError, TypeError, AttributeError):
                            # Skip sensors that can't be parsed
                            pass

            # We need at least weight to proceed
            if "weight" not in measurements:
                _LOGGER.warning(
                    "Cannot reassign from user %s: weight sensor not found or unavailable",
                    from_user_id,
                )
                return False

            _LOGGER.debug(
                "Retrieved %d raw measurements from sensor states for user %s",
                len(measurements),
                from_user_id,
            )
        else:
            # Even if we have measurements in memory, only use raw measurements
            # Filter to only weight and impedance (body metrics will be recalculated)
            raw_measurements = {}
            if "weight" in measurements:
                raw_measurements["weight"] = measurements["weight"]
            if "impedance" in measurements:
                raw_measurements["impedance"] = measurements["impedance"]
            measurements = raw_measurements

            if "weight" not in measurements:
                _LOGGER.warning(
                    "Cannot reassign from user %s: no weight measurement available",
                    from_user_id,
                )
                return False

        # Validate target user exists
        if to_user_id not in self._user_profiles_by_id:
            _LOGGER.error("Target user %s not found in user profiles", to_user_id)
            return False

        _LOGGER.debug(
            "Reassigning raw measurement from user %s to user %s (weight: %.2f kg%s)",
            from_user_id,
            to_user_id,
            measurements.get("weight"),
            f", impedance: {measurements.get('impedance')} Ω"
            if "impedance" in measurements
            else "",
        )

        # Create ScaleData with only raw measurements
        # Body metrics will be recalculated by _route_to_user() based on target user's profile
        scale_data = ScaleData(measurements=measurements)

        # Route to target user (this will store in _last_user_measurement for target)
        self._route_to_user(to_user_id, scale_data)

        # Signal source user's sensors to revert to previous using direct callback registry
        revert_data = ScaleData(measurements={"_revert_": True})
        for update_callback in self._user_callbacks.get(from_user_id, []):
            try:
                update_callback(revert_data)
            except Exception as ex:
                _LOGGER.error(
                    "Error reverting sensor for user %s: %s", from_user_id, ex
                )

        # Clear source user's measurement from tracking to prevent stale data
        self._last_user_measurement.pop(from_user_id, None)
        _LOGGER.debug(
            "Cleared measurement tracking for source user %s after reassignment",
            from_user_id,
        )

        return True

    def remove_user_measurement(self, user_id: str) -> bool:
        """Remove a user's last measurement (revert to previous or unavailable).

        Args:
            user_id: The user ID to remove the measurement from.

        Returns:
            True if removal succeeded, False otherwise.
        """
        _LOGGER.debug("Removing last measurement for user %s", user_id)

        # Remove from tracking (if present)
        self._last_user_measurement.pop(user_id, None)

        # Signal user's sensors to revert using direct callback registry
        revert_data = ScaleData(measurements={"_revert_": True})
        for update_callback in self._user_callbacks.get(user_id, []):
            try:
                update_callback(revert_data)
            except Exception as ex:
                _LOGGER.error("Error reverting sensor for user %s: %s", user_id, ex)

        return True
