"""Coordinator for the etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import asyncio
import logging
import platform
from collections.abc import Callable
from datetime import date
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    ESF24Scale,
    ESF551Scale,
    EtekcitySmartFitnessScale,
    EtekcitySmartFitnessScaleWithBodyMetrics,
    ScaleData,
    Sex,
    WeightUnit,
)
from habluetooth import HaScannerRegistration
from homeassistant.core import HomeAssistant, callback
from .const import ScaleModel

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
        detection_callback: Optional[Callable[[BLEDevice, AdvertisementData], None]],
        service_uuids: Optional[List[str]],
        scanning_mode: Literal["active", "passive"],
        clients: List[APIClient],
        **kwargs,
    ):
        """
        Initialize the ESPHome scanner.

        Args:
            detection_callback: Function called when a device advertisement is detected.
            service_uuids: Optional list of service UUIDs to filter advertisements.
            scanning_mode: Whether to use active or passive scanning.
            clients: List of ESPHome API clients to use as Bluetooth proxies.
            **kwargs: Additional arguments (not used).
        """
        super().__init__(detection_callback, service_uuids)

        self._clients = list(clients)
        self._scanning = False

        # Per-client tracking
        self._client_info: Dict[APIClient, Optional[DeviceInfo]] = {
            client: None for client in self._clients
        }
        self._client_features: Dict[APIClient, int] = {
            client: 0 for client in self._clients
        }
        self._client_unsubscribers: Dict[APIClient, Optional[Callable[[], None]]] = {
            client: None for client in self._clients
        }
        self._active_clients: Dict[APIClient, Dict[str, Any]] = {}

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
        detection_callback: Optional[Callable[[BLEDevice, AdvertisementData], None]],
        service_uuids: Optional[List[str]],
        scanning_mode: Literal["active", "passive"],
        clients: List[APIClient],
        adapter: str | None = None,
        **kwargs,
    ):
        """
        Initialize the hybrid scanner.

        Args:
            detection_callback: Function called when a device advertisement is detected.
            service_uuids: Optional list of service UUIDs to filter advertisements.
            scanning_mode: Whether to use active or passive scanning.
            clients: List of ESPHome API clients to use as Bluetooth proxies.
            adapter: The Bluetooth adapter to use for native scanning (Linux only).
            **kwargs: Additional arguments passed to the native scanner.
        """
        super().__init__(None, service_uuids)

        self._native_scanner = None
        self._proxy_scanner = None
        self._scanners: List[BaseBleakScanner] = []
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
        self, callback: Optional[AdvertisementDataCallback]
    ) -> Callable[[], None]:
        for scanner in self._scanners:
            try:
                scanner.register_detection_callback(callback)
            except Exception as ex:
                _LOGGER.exception(
                    f"Error registering detection callback on {type(scanner).__name__}: {ex}"
                )

    @property
    def seen_devices(self) -> Dict[str, Tuple[BLEDevice, AdvertisementData]]:
        """Get the dictionary of seen devices."""
        seen: Dict[str, Tuple[BLEDevice, AdvertisementData]] = {}

        for scanner in self._scanners:
            seen |= scanner.seen_devices

        return seen

    @seen_devices.setter
    def seen_devices(
        self, value: Dict[str, Tuple[BLEDevice, AdvertisementData]]
    ) -> None:
        """Set the dictionary of seen devices."""
        # This is intentionally a no-op as we don't want to override
        # the seen devices of individual scanners
        pass


class ScaleDataUpdateCoordinator:
    """
    Coordinator to manage data updates for a scale device.

    This class handles the communication with the Etekcity Smart Fitness Scale
    and coordinates updates to the Home Assistant entities.
    """

    _client: Optional[EtekcitySmartFitnessScale] = None
    _display_unit: Optional[WeightUnit] = None
    _scanner_change_cb_unregister: Optional[Callable[[], None]] = None

    body_metrics_enabled: bool = False
    _sex: Optional[Sex] = None
    _birthdate: Optional[date] = None
    _height_m: Optional[float] = None

    def __init__(
        self,
        hass: HomeAssistant,
        address: str,
        scale_model: ScaleModel = ScaleModel.ESF551,
    ) -> None:
        """Initialize the ScaleDataUpdateCoordinator.

        Args:
            hass: The Home Assistant instance.
            address: The Bluetooth address of the scale.
            scale_model: The detected scale model.
        """
        self.address = address
        self._hass = hass
        self._scale_model = scale_model
        self._lock = asyncio.Lock()
        self._listeners: Dict[Callable[[], None], Callable[[ScaleData], None]] = {}

    def set_display_unit(self, unit: WeightUnit) -> None:
        """Set the display unit for the scale.

        Args:
            unit: The weight unit to display on the scale.
        """
        _LOGGER.debug("Setting display unit to: %s", unit.name)
        self._display_unit = unit
        if self._client:
            self._client.display_unit = unit

    async def _get_bluetooth_scanner(self) -> Optional[BaseBleakScanner]:
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
            esphome_clients: List[APIClient] = []
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
            scanner: Optional[BaseBleakScanner] = None
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

            # Initialize appropriate client based on scale model
            try:
                if self.body_metrics_enabled:
                    if (
                        self._sex is None
                        or self._birthdate is None
                        or self._height_m is None
                    ):
                        _LOGGER.error(
                            "Body metrics enabled but required parameters are missing"
                        )
                        raise ValueError("Missing required body metrics parameters")

                    # Body metrics only supported on ESF-551
                    if self._scale_model != ScaleModel.ESF551:
                        _LOGGER.warning(
                            "Body metrics requested but scale model %s does not support body metrics. "
                            "Disabling body metrics.",
                            self._scale_model,
                        )
                        self.body_metrics_enabled = False
                    else:
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
                            bleak_scanner_backend=scanner,
                        )
                else:
                    # Choose appropriate scale class based on model
                    if self._scale_model == ScaleModel.ESF24:
                        _LOGGER.debug(
                            "Initializing new ESF24Scale client (experimental)"
                        )
                        self._client = ESF24Scale(
                            self.address,
                            self.update_listeners,
                            self._display_unit,
                            bleak_scanner_backend=scanner,
                        )
                    elif self._scale_model == ScaleModel.ESF551:
                        _LOGGER.debug("Initializing new ESF551Scale client")
                        self._client = ESF551Scale(
                            self.address,
                            self.update_listeners,
                            self._display_unit,
                            bleak_scanner_backend=scanner,
                        )
                    else:
                        # Fallback to ESF551 for backward compatibility
                        _LOGGER.debug(
                            "Unknown scale model, defaulting to ESF551Scale client"
                        )
                        self._client = ESF551Scale(
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

    @callback
    def update_listeners(self, data: ScaleData) -> None:
        """Update all registered listeners with improved logging.

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

        # Update all listeners
        listener_count = len(self._listeners)
        _LOGGER.debug("Updating %d listeners with new scale data", listener_count)

        # Make a copy of listeners to avoid modification during iteration
        for update_callback in list(self._listeners.values()):
            try:
                update_callback(data)
            except Exception as ex:
                _LOGGER.error("Error updating listener: %s", ex)

    async def enable_body_metrics(
        self, sex: Sex, birthdate: date, height_m: float
    ) -> None:
        """Enable body metrics calculations.

        Args:
            sex: The sex of the user.
            birthdate: The birthdate of the user.
            height_m: The height of the user in meters.
        """
        if not self.body_metrics_enabled:
            _LOGGER.debug(
                "Enabling body metrics with sex=%s, birthdate=%s, height=%f",
                sex,
                birthdate,
                height_m,
            )

            async with self._lock:
                self.body_metrics_enabled = True
                self._sex = sex
                self._birthdate = birthdate
                self._height_m = height_m

                if self._client:
                    try:
                        await self._async_start()
                    except Exception as ex:
                        _LOGGER.error(
                            "Failed to restart client after enabling body metrics: %s",
                            ex,
                        )

    async def disable_body_metrics(self) -> None:
        """Disable body metrics calculations."""
        if self.body_metrics_enabled:
            _LOGGER.debug("Disabling body metrics")

            async with self._lock:
                self.body_metrics_enabled = False
                self._sex = None
                self._birthdate = None
                self._height_m = None

                if self._client:
                    try:
                        await self._async_start()
                    except Exception as ex:
                        _LOGGER.error(
                            "Failed to restart client after disabling body metrics: %s",
                            ex,
                        )
