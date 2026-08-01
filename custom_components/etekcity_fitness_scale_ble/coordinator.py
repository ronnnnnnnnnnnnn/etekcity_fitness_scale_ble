"""Coordinator for the etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from copy import deepcopy
from datetime import datetime, timedelta
import logging
from math import floor
import platform
import time
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NamedTuple
from urllib.parse import quote

from aioesphomeapi import APIClient, BluetoothProxyFeature
from aioesphomeapi.api_pb2 import (  # type: ignore[attr-defined]
    BluetoothLEAdvertisementResponse,
    BluetoothLERawAdvertisementsResponse,
)
from aioesphomeapi.model import APIVersion, BluetoothLEAdvertisement, DeviceInfo
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
    SCALE_CLASSES,
    BluetoothScanningMode,
    EtekcitySmartFitnessScale,
    ScaleData,
    WeightUnit,
)
from habluetooth import HaScannerRegistration
from habluetooth import get_manager as habluetooth_get_manager
from homeassistant.core import CoreState, HomeAssistant, callback
from homeassistant.components import persistent_notification
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH
from homeassistant.helpers.event import async_call_later
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED, UnitOfMass
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import MassConverter

from .const import (
    CONF_ATHLETE,
    CONF_ENABLE_LIBRARY_LOGGING,
    CONF_HISTORY_RETENTION_DAYS,
    CONF_KEEP_HISTORY_FOREVER,
    CONF_MAX_HISTORY_SIZE,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_USER_ID,
    CONF_USER_NAME,
    CONF_WEIGHT_HISTORY,
    DOMAIN,
    HISTORY_RETENTION_DAYS,
    MAX_HISTORY_SIZE,
    PASSIVE_SCAN_ISSUE_ID,
    ScaleModel,
    parse_notify_service,
)
from .person_detector import PersonDetector

SYSTEM = platform.system()
IS_LINUX = SYSTEM == "Linux"
IS_MACOS = SYSTEM == "Darwin"


if IS_LINUX:
    from bleak.args.bluez import BlueZScannerArgs, OrPattern

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


# Backoff for scale-client restarts triggered by BT-scanner-registration
# changes (see `_async_registration_changed`). Without this, a scanner that
# keeps failing to (re)start gets a full pipeline rebuild attempt on every
# single registration event (e.g. each ESPHome proxy reconnect), which can
# run often enough to saturate the HA event loop for hours.
RESTART_BACKOFF_BASE_SECONDS = 30
RESTART_BACKOFF_MAX_SECONDS = 1800  # 30 minutes


# Hard floor between scale-client restart attempts triggered by a
# registration event, measured from the *end* of the previous attempt
# (see `_async_registration_changed`). HA's
# scanner-registration callback isn't itself rate-limited, so without this
# floor a flapping/bootlooping BT proxy could drive back-to-back
# `_async_start()` cycles - no leak (that's fixed), but repeated real I/O
# load faster than the backoff is meant to allow. Applies unconditionally
# to every restart attempt made from a registration event, regardless of
# whether a backoff retry happens to be pending - it does not touch the
# backoff delay itself. Events landing inside the floor are deferred, not
# dropped: they coalesce into a single retry scheduled for the floor's
# expiry, so no registration event is ever lost.
REGISTRATION_PREEMPT_DEBOUNCE_SECONDS = 5


_LOGGER = logging.getLogger(__name__)


class _NoiseFilter(logging.Filter):
    """Drop ``BleakOutOfConnectionSlotsError`` records from the library logger.

    The etekcity_esf551_ble library logs *any* connect failure via
    ``logger.exception(...)`` (ERROR + traceback). The failure most users
    will ever see is this one exception, raised when the scale is
    advertising but not connectable — typically its post-measurement
    spin-down tail. The GATT models default to no cooldown, so every
    straggler advertisement in that tail starts a fresh connect cycle that
    ends this way. The integration recovers on the next real advertisement,
    so the log line is just noise; genuine connection-slot exhaustion isn't
    actionable from a log line either.

    Bypassed when ``enable_library_logging`` is on, so debug sessions still
    see every connect attempt.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None and type(exc).__name__ == "BleakOutOfConnectionSlotsError":
            return False
        # Belt-and-suspenders for records carrying the name but no exc_info.
        return "BleakOutOfConnectionSlotsError" not in record.getMessage()


def _norm_country_code(config: Any) -> str:
    """Extract normalized 2-letter country code from HA config."""
    raw = (getattr(config, "country", None) or "").upper()
    return raw[:2] if len(raw) >= 2 else ""


class BluetoothNotAvailableError(Exception):
    """Exception raised when no Bluetooth adapter or ESPHome proxy is available."""

    pass


def _get_bluetooth_manager() -> Any | None:
    """Return HA's shared BluetoothManager, or None if unavailable.

    Resolved via habluetooth's process-global accessor, which the bluetooth
    integration populates eagerly (``set_manager``) during its
    ``async_setup`` — i.e. before this integration's entries run
    (``after_dependencies``). This is the authoritative source: HA's own
    ``bluetooth.api`` derives its lazily-populated
    ``hass.data["bluetooth_manager"]`` cache from it, so that key can lag
    behind (absent on e.g. proxy-only installs until something touches the
    api module — the trap the previous hass.data-based lookup fell into)
    while the accessor is already valid.
    """
    try:
        return habluetooth_get_manager()
    except Exception:  # noqa: BLE001 - RuntimeError when not yet set
        return None


class _BluetoothTopology(NamedTuple):
    """The slice of HA's Bluetooth topology this integration is built on.

    Used as the snapshot for relevance-filtering scanner-registration
    events (see `_topology_unchanged_since_last_start`). Scanner sources
    belonging to other integrations (e.g. Shelly BLE proxies) are
    deliberately not represented.
    """

    native: bool
    native_passive: bool
    native_sources: frozenset
    # bleak-esphome ESPHomeClientData objects; held strongly so identity
    # (`is`) comparison against a later walk is sound.
    esphome_client_data: tuple


class _TopologyScan(NamedTuple):
    """Result of a `_scan_bluetooth_topology` walk."""

    # False when native-adapter detection errored - the walk's topology is
    # then untrustworthy and must never be used to suppress restarts.
    adapter_check_ok: bool
    topology: _BluetoothTopology


class BleakScannerESPHome(BaseBleakScanner):
    """
    A BLE scanner implementation that uses ESPHome devices as Bluetooth proxies.

    This scanner piggybacks on Home Assistant's existing ESPHome API
    connections to receive Bluetooth advertisements. This allows for extended
    range and coverage compared to a single local Bluetooth adapter.

    The scanner is deliberately wire-silent on those shared connections: it
    never sends Subscribe/UnsubscribeBluetoothLEAdvertisementsRequest. HA's
    esphome integration (bleak-esphome's ``connect_scanner``) has already
    subscribed before the proxy's scanner source is even registered, the
    firmware allows only one advertisement subscription per device anyway
    ("Only one API subscription is allowed at a time"), and an Unsubscribe
    sent on the shared connection would tear down HA's own advertisement
    stream for every integration. Instead, a message callback is registered
    directly on the live ``APIConnection`` — aioesphomeapi dispatches
    incoming messages by message *type* to every handler registered on the
    connection, regardless of which subscription elicited them — and only
    that local handler is removed on stop.
    """

    def __init__(
        self,
        detection_callback: Callable[[BLEDevice, AdvertisementData], None] | None,
        service_uuids: list[str] | None,
        scanning_mode: Literal["active", "passive"],
        clients: list[Any],
        **kwargs,
    ):
        """
        Initialize the ESPHome scanner.

        Args:
            detection_callback: Function called when a device advertisement is detected.
            service_uuids: Optional list of service UUIDs to filter advertisements.
            scanning_mode: Whether to use active or passive scanning.
            clients: list of bleak-esphome ``ESPHomeClientData``-shaped objects
                (duck-typed: ``.client`` (APIClient), ``.device_info``
                (DeviceInfo), ``.api_version`` (APIVersion)) for the ESPHome
                devices to use as Bluetooth proxies. Carrying the cached
                DeviceInfo/APIVersion avoids any wire round-trip at start()
                (a ``device_info()`` call here has been seen to time out in
                the field — issue #38).
            **kwargs: Additional arguments (not used).
        """
        super().__init__(detection_callback, service_uuids)

        # ESPHomeClientData is an unhashable slots dataclass (eq=True, no
        # __hash__) and DeviceInfo contains lists, so all per-client state
        # stays keyed by the identity-hashable APIClient.
        self._clients: list[APIClient] = [cd.client for cd in clients]
        self._scanning = False

        # Per-client tracking, pre-seeded from HA's cached connection data
        self._client_info: dict[APIClient, DeviceInfo | None] = {
            cd.client: cd.device_info for cd in clients
        }
        self._client_api_version: dict[APIClient, APIVersion | None] = {
            cd.client: cd.api_version for cd in clients
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

        # Initialize all clients. No wire calls happen here: device info and
        # API version were captured from HA's own connection data at
        # construction time (probing the device again with device_info() is
        # redundant and has been seen to time out in the field — issue #38).
        for client in self._clients:
            try:
                # A real APIClient has no public is_connected; the live
                # connection object is what we register callbacks on, so its
                # absence is the definitive "not connected" signal.
                if getattr(client, "_connection", None) is None:
                    _LOGGER.warning(
                        "Client %s is not connected, skipping", client.address
                    )
                    continue

                # Detect Bluetooth features from the cached device info
                self._client_features[client] = self._detect_bluetooth_features(client)

                if not self._client_features[client]:
                    # Mirrors HA itself, which deregisters proxies reporting
                    # no Bluetooth feature flags.
                    _LOGGER.warning(
                        "Client %s does not support Bluetooth proxy, skipping",
                        client.address,
                    )
                    continue

                self._active_clients[client] = {
                    "name": self._client_info[client].name,
                    "features": self._client_features[client],
                }

                # Register the local advertisement callback with error handling
                try:
                    self._subscribe_to_advertisements(client)
                    successful_clients += 1
                except Exception as ex:
                    _LOGGER.error(
                        "Failed to subscribe to advertisements for %s: %s",
                        client.address,
                        ex,
                    )
                    self._active_clients.pop(client, None)
                    continue

                _LOGGER.debug(
                    "Client %s initialized with features: %s",
                    self._client_info[client].name,
                    self._client_features[client],
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
        """Stop scanning for devices.

        Only removes the locally registered message callbacks — nothing is
        sent on the wire. In particular, no
        UnsubscribeBluetoothLEAdvertisementsRequest goes out on HA's shared
        connections, which would kill HA's own advertisement stream for the
        proxy (the firmware tracks a single subscriber slot per device).
        """
        if not self._scanning:
            return

        # Deregister the local callbacks from all clients
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

    def _on_bluetooth_le_advertisement_response(self, client: APIClient, msg) -> None:
        """Adapt a raw protobuf advertisement response to the model object.

        Registered directly on the APIConnection (see
        `_subscribe_to_advertisements`), so the payload is the protobuf
        message rather than the BluetoothLEAdvertisement model that
        aioesphomeapi's own subscribe helper would have produced.
        """
        try:
            adv = BluetoothLEAdvertisement.from_pb(msg)
        except Exception:
            # Must never leak into aioesphomeapi's packet-processing path:
            # on older versions an exception there tears down HA's shared
            # ESPHome connection (isolated upstream since aioesphomeapi
            # #1755, but cheap to guard here).
            _LOGGER.exception(
                "Error parsing advertisement response from %s", client.address
            )
            return
        self._on_bluetooth_le_advertisement(client, adv)

    def _on_bluetooth_le_advertisement(
        self, client: APIClient, adv: BluetoothLEAdvertisement
    ) -> None:
        """Handle a Bluetooth LE advertisement from a specific client."""
        try:
            self._handle_bluetooth_le_advertisement(client, adv)
        except Exception:
            # See _on_bluetooth_le_advertisement_response: handler exceptions
            # must not propagate into the connection's packet processing.
            _LOGGER.exception("Error handling advertisement from %s", client.address)

    def _handle_bluetooth_le_advertisement(
        self, client: APIClient, adv: BluetoothLEAdvertisement
    ) -> None:
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
        try:
            self._handle_bluetooth_le_raw_advertisement(client, response)
        except Exception:
            # See _on_bluetooth_le_advertisement_response: handler exceptions
            # must not propagate into the connection's packet processing.
            _LOGGER.exception(
                "Error handling raw advertisements from %s", client.address
            )

    def _handle_bluetooth_le_raw_advertisement(
        self, client: APIClient, response
    ) -> None:
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
        """Register a LOCAL advertisement callback on HA's live connection.

        Deliberately does NOT call the client's subscribe_bluetooth_le_*
        helpers: those send SubscribeBluetoothLEAdvertisementsRequest (the
        firmware rejects it — HA already holds the device's single
        subscriber slot) and their unsubscribers send
        UnsubscribeBluetoothLEAdvertisementsRequest, which would tear down
        HA's own advertisement stream. `APIConnection.add_message_callback`
        registers the handler with no wire traffic, and its returned remover
        is local-only. The message type must match what HA's subscription
        elicits, so the raw/processed choice mirrors bleak-esphome's own
        (`RAW_ADVERTISEMENTS` feature flag on the same cached DeviceInfo).
        """
        features = self._client_features[client]
        connection = client._connection  # noqa: SLF001

        if features & BluetoothProxyFeature.RAW_ADVERTISEMENTS:
            self._client_unsubscribers[client] = connection.add_message_callback(
                partial(self._on_bluetooth_le_raw_advertisement, client),
                (BluetoothLERawAdvertisementsResponse,),
            )
            _LOGGER.debug("%s: Listening for raw advertisements", client.address)
        else:
            self._client_unsubscribers[client] = connection.add_message_callback(
                partial(self._on_bluetooth_le_advertisement_response, client),
                (BluetoothLEAdvertisementResponse,),
            )
            _LOGGER.debug("%s: Listening for processed advertisements", client.address)

    def _detect_bluetooth_features(self, client: APIClient) -> int:
        """Detect supported Bluetooth features for a specific client."""
        device_info = self._client_info.get(client)
        if not device_info:
            return 0

        # Check if the device info has the bluetooth_proxy_feature_flags_compat method
        if hasattr(device_info, "bluetooth_proxy_feature_flags_compat"):
            # Use the API version cached alongside the device info —
            # client.api_version is None while disconnected, which would
            # raise TypeError inside the compat comparison.
            api_version = self._client_api_version.get(client)
            if api_version is None:
                return 0
            return device_info.bluetooth_proxy_feature_flags_compat(api_version)

        # Fallback detection based on features list
        features = device_info.features if device_info else []
        if any("bluetooth" in feature.lower() for feature in features):
            # Assume basic support if "bluetooth" is mentioned in features
            return BluetoothProxyFeature.ACTIVE_CONNECTIONS

        return 0


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
        clients: list[Any],
        adapter: str | None = None,
        **kwargs,
    ):
        """
        Initialize the hybrid scanner.

        Args:
            detection_callback: Function called when a device advertisement is detected.
            service_uuids: Optional list of service UUIDs to filter advertisements.
            scanning_mode: Whether to use active or passive scanning.
            clients: list of bleak-esphome ``ESPHomeClientData``-shaped
                objects for the ESPHome proxies (see BleakScannerESPHome).
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
            PlatformBleakScanner, _ = get_platform_scanner_backend_type()
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
        """Start scanning for devices.

        Tolerates a partially-failing scanner instead of failing the whole
        hybrid start. Previously a bare ``asyncio.gather`` here meant that if
        one scanner's ``start()`` raised (e.g. native passive scanning
        unavailable: "passive scanning on Linux requires BlueZ >= 5.56 with
        --experimental enabled"), the exception propagated up while the
        *other* scanner (e.g. the ESPHome proxy) had already finished
        starting — but because ``self._scanning`` was never set to ``True``,
        the ``except`` branch's call to ``self.stop()`` was a no-op (``stop``
        early-returned when ``_scanning`` was falsy). The already-started
        scanner was never stopped, leaking a live subscription. Each
        subsequent registration-change restart (see
        ``_async_registration_changed``) constructed a fresh
        ``BleakScannerHybrid`` on top of the still-running leaked one, so
        every incoming advertisement batch was processed once per leaked
        instance — after enough restarts this alone saturated the event
        loop, independent of restart frequency.

        Now each scanner starts independently; a scanner whose ``start()``
        raises is stopped explicitly and dropped, and scanning continues
        with whichever scanners did start. Only if none of them start do we
        raise.
        """
        if self._scanning:
            return

        if not self._scanners:
            raise BleakError("No scanners available")

        # Start all scanners concurrently, collecting failures instead of
        # letting the first one abort everything.
        results = await asyncio.gather(
            *[scanner.start() for scanner in self._scanners],
            return_exceptions=True,
        )

        started: list[BaseBleakScanner] = []
        for scanner, result in zip(self._scanners, results):
            if isinstance(result, BaseException):
                _LOGGER.warning(
                    "Failed to start %s (%s); continuing without it",
                    type(scanner).__name__,
                    result,
                )
                # The scanner's own start() failed, but it may still have
                # partially initialized (e.g. subscribed to advertisements)
                # before raising. Stop it explicitly here rather than via
                # self.stop() — self._scanners still holds every scanner in
                # this loop, so self.stop() would also stop scanners that
                # already started successfully in this same pass.
                try:
                    await scanner.stop()
                except Exception:  # noqa: BLE001
                    pass
                if scanner is self._native_scanner:
                    self._native_scanner = None
                if scanner is self._proxy_scanner:
                    self._proxy_scanner = None
            else:
                started.append(scanner)

        self._scanners = started

        if not started:
            _LOGGER.error("Error starting hybrid scanner: no scanner could start")
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

    async def stop(self) -> None:
        """Stop scanning for devices.

        Stops every scanner in ``self._scanners`` unconditionally, not
        gated behind ``self._scanning``. A hybrid that never finished
        starting (e.g. ``self._scanning`` still ``False`` for any reason
        other than the already-handled partial-failure path in
        ``start()``) must still have its children stopped — otherwise a
        live, subscribed scanner could again be stranded, as happened
        before ``start()`` was fixed to clean up after itself.
        """
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
        scale_model: ScaleModel = ScaleModel.ESF551,
    ) -> None:
        """Initialize the ScaleDataUpdateCoordinator.

        Args:
            hass: The Home Assistant instance.
            address: The Bluetooth address of the scale.
            user_profiles: list of user profile dictionaries.
            device_name: The device name used for entity ID construction.
            scale_model: The detected scale model.
        """
        self.address = address
        self._hass = hass
        self._device_name = device_name
        self._scale_model = scale_model
        self._lock = asyncio.Lock()
        # Set once by `async_stop`, under `_lock`. Lets a
        # `_async_registration_changed` call that was already scheduled
        # before the stop (e.g. a registration event that fired just
        # before unload) notice, once it's its turn at the lock, that the
        # coordinator is gone and bail out instead of building a new scale
        # client after shutdown. Never reset - each config-entry setup
        # creates a fresh coordinator instance.
        self._stopped = False
        # Backoff state for scanner-change restarts (see
        # `_async_registration_changed`). Counts consecutive `_async_start`
        # failures and holds the cancel handle of the (single, coalesced)
        # scheduled retry.
        self._restart_failures = 0
        self._restart_retry_unsub: Callable[[], None] | None = None
        # Monotonic timestamp of the end of the last `_async_start` attempt
        # made from `_async_registration_changed`, used to enforce
        # `REGISTRATION_PREEMPT_DEBOUNCE_SECONDS` on the pre-emption path.
        self._last_restart_attempt_monotonic: float | None = None
        # Relevance filter for scanner-registration events (both only
        # touched under `_lock`, like the rest of the restart state).
        # `_pending_topology` is the topology snapshot captured by the
        # `_get_bluetooth_scanner` walk the scanner was built from;
        # `_last_topology` is that snapshot promoted once the client
        # actually started, and None whenever no client is running or the
        # last start failed (None always lets a restart proceed). Holding
        # the ESPHomeClientData objects strongly is deliberate: identity
        # comparison against them detects a proxy that bounced (bleak-esphome
        # builds a new client_data per connect) even when the source name
        # is unchanged, and the held references make `is` checks sound.
        self._pending_topology: _BluetoothTopology | None = None
        self._last_topology: _BluetoothTopology | None = None
        # Scanning mode for the library's own fallback scanner, used only
        # when `_get_bluetooth_scanner` returns None (native adapter, no
        # ESPHome proxies). PASSIVE coexists with HA's shared scanner;
        # `_get_bluetooth_scanner` downgrades it to ACTIVE when the adapter
        # can't do passive scanning.
        self._fallback_scanning_mode = BluetoothScanningMode.PASSIVE
        # One-shot EVENT_HOMEASSISTANT_STARTED listener deferring the first
        # ACTIVE-fallback start until HA has fully started (see
        # `_async_start`). Only ever set when that fallback would otherwise
        # fire into the boot window; cancelled in `async_stop`.
        self._started_listener_unsub: Callable[[], None] | None = None
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

    # Countries that typically use 12-hour time (HA backend does not expose
    # per-user time format; use country as heuristic).
    _COUNTRY_12H: frozenset[str] = frozenset({"US", "CA", "PH", "AU"})

    def _get_display_preferences(self) -> tuple[str, str | None, str | None]:
        """Return (language, time_format, date_format) from HA config.

        time_format: '12' | '24' | None. If not set on config, inferred from
            hass.config.country (12h for US, CA, PH, AU; 24h otherwise).
        date_format: 'dmy' | 'mdy' | 'ymd' | None only when explicitly set;
            we do not infer from country so that we can use an unambiguous
            spelled-out date (e.g. "Mar 8, 2026") when date_format is None.
        """
        config = getattr(self._hass, "config", None)
        if config is None:
            return "en", None, None
        language = getattr(config, "language", None) or "en"
        time_fmt = getattr(config, "time_format", None)
        if time_fmt in ("language", "auto", ""):
            time_fmt = None
        date_fmt = getattr(config, "date_format", None)
        if date_fmt in ("language", "auto", ""):
            date_fmt = None
        country = _norm_country_code(config)
        if time_fmt is None and country:
            time_fmt = "12" if country in self._COUNTRY_12H else "24"
        return language, time_fmt, date_fmt

    def _format_time_part(
        self, localized: datetime, time_format: str | None, include_seconds: bool
    ) -> str | None:
        """Format time part from display preferences. Returns None if Babel should be used."""
        if time_format == "24":
            fmt = "%H:%M:%S" if include_seconds else "%H:%M"
            return localized.strftime(fmt)
        if time_format == "12":
            fmt = "%I:%M:%S %p" if include_seconds else "%I:%M %p"
            return localized.strftime(fmt).lstrip("0")
        return None

    def _format_date_unambiguous(self, localized: datetime, language: str) -> str:
        """Format date with spelled month (e.g. Mar 8, 2026) for clarity."""
        try:
            from babel.dates import format_date as babel_format_date

            return babel_format_date(
                localized,
                format="medium",
                locale=language.replace("-", "_"),
            )
        except Exception as err:
            _LOGGER.debug(
                "Babel format_date failed (locale=%s), using strftime fallback: %s",
                language,
                err,
            )
            return localized.strftime("%b %d, %Y")

    def _format_notification_timestamp(self, timestamp_str: str) -> str:
        """Format timestamp for display in notifications."""
        try:
            localized = dt_util.as_local(datetime.fromisoformat(timestamp_str))
        except (ValueError, TypeError):
            return timestamp_str
        language, time_format, date_format = self._get_display_preferences()
        if date_format is not None:
            date_patterns = {
                "dmy": "%d/%m/%Y",
                "mdy": "%m/%d/%Y",
                "ymd": "%Y-%m-%d",
            }
            date_part = date_patterns.get((date_format or "").lower(), "%b %d, %Y")
            time_fmt = "%H:%M:%S" if time_format == "24" else "%I:%M:%S %p"
            return localized.strftime(f"{date_part} at {time_fmt} %Z")
        time_str = self._format_time_part(localized, time_format, True)
        if time_str is None:
            time_str = localized.strftime("%I:%M:%S %p").lstrip("0")
        date_str = self._format_date_unambiguous(localized, language)
        return f"{date_str} at {time_str} {localized.strftime('%Z')}"

    def _format_notification_time(self, timestamp_str: str) -> str:
        """Format time for mobile notifications."""
        try:
            localized = dt_util.as_local(datetime.fromisoformat(timestamp_str))
        except (ValueError, TypeError):
            return timestamp_str
        language, time_format, _date_fmt = self._get_display_preferences()
        time_str = self._format_time_part(localized, time_format, False)
        if time_str is not None:
            return time_str
        try:
            from babel.dates import format_datetime as babel_format_datetime

            return babel_format_datetime(
                localized, format="short", locale=language.replace("-", "_")
            )
        except Exception as err:
            _LOGGER.debug(
                "Babel format_datetime failed (locale=%s), using strftime fallback: %s",
                language,
                err,
            )
            return localized.strftime("%I:%M %p").lstrip("0")

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
        # Normalize all measurements to ensure consistent field order, skipping
        # entries with a missing or non-numeric weight_kg: one corrupted entry
        # must not raise on every state write and wedge the sensor.
        normalized = []
        for m in history:
            weight_kg = m.get("weight_kg")
            if isinstance(weight_kg, bool) or not isinstance(weight_kg, (int, float)):
                _LOGGER.warning(
                    "Skipping history entry with invalid weight_kg %r for user %s (timestamp: %s)",
                    weight_kg,
                    user_id,
                    m.get("timestamp"),
                )
                continue
            normalized.append(self._normalize_measurement(m))
        return normalized

    def get_user_history_for_display(
        self, user_id: str, display_unit: str | None = None
    ) -> list[dict]:
        """Get weight history formatted for display with user-friendly keys.

        Converts weight to the given display unit and uses friendly key names.

        Args:
            user_id: The user ID to get history for.
            display_unit: Target mass unit (a ``UnitOfMass`` value), e.g. the
                unit the calling entity's state is displayed in. When None,
                falls back to the scale-LCD display unit configured for the
                device.

        Returns:
            List of measurement dicts formatted for display with keys:
            - "Timestamp" (instead of "timestamp")
            - "Weight (kg)" / "Weight (lbs)" / "Weight (<unit>)"
              (instead of "weight_kg")
            - "Impedance (Ω)" (instead of "impedance_ohm")
        """
        # The recorder stores {} for ALL of a state's attributes once they
        # exceed MAX_STATE_ATTRS_BYTES (16 KB; ~85 bytes/entry → ~190 entries),
        # and max_history_size is user-configurable up to 1000 — this cap must
        # not be removed.
        history = self.get_user_history(user_id)[-20:]
        if display_unit is None:
            display_unit = (
                UnitOfMass.POUNDS
                if self.get_display_unit() == WeightUnit.LB
                else UnitOfMass.KILOGRAMS
            )
        # Keep the historical key spellings for kg/lb (backward compatibility);
        # any other mass unit gets a generic "Weight (<unit>)" key.
        if display_unit == UnitOfMass.KILOGRAMS:
            weight_key = "Weight (kg)"
        elif display_unit == UnitOfMass.POUNDS:
            weight_key = "Weight (lbs)"
        else:
            weight_key = f"Weight ({display_unit})"

        display_history = []
        for measurement in history:
            display_measurement = {}
            # Timestamp with friendly key
            display_measurement["Timestamp"] = measurement["timestamp"]

            # Weight with friendly key, converted to the display unit
            display_measurement[weight_key] = round(
                MassConverter.convert(
                    measurement["weight_kg"], UnitOfMass.KILOGRAMS, display_unit
                ),
                2,
            )

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

    def _get_keep_history_forever(self) -> bool:
        """Check if the user disabled automatic history cleanup.

        Returns:
            True when age/size limits must not prune history (default: False).
        """
        if not self._config_entry_id:
            return False

        entry = self._hass.config_entries.async_get_entry(self._config_entry_id)
        if not entry:
            return False

        return entry.data.get(CONF_KEEP_HISTORY_FOREVER, False)

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

    def _configure_library_logger(self) -> logging.Logger:
        """Logger handed to the etekcity_esf551_ble client.

        Always a child of this integration's logger, so library output lands in
        our namespace: captured by HA's per-integration debug toggle, included
        in diagnostics, and attributable to this device rather than to a
        separate library tree.

        When the advanced option is on we pin the child to DEBUG so
        protocol-level frames show up regardless of the integration's own log
        level. Otherwise the library logs at the integration's level, so HA's
        per-integration "Enable debug logging" button reaches it too — pinning
        a floor here would override that button and leave it producing nothing
        from the library.

        Also installs ``_NoiseFilter`` to drop ``BleakOutOfConnectionSlotsError``
        records, which the library logs at ERROR with a traceback whenever the
        scale advertises while not connectable. Verbose logging bypasses the
        filter so debug sessions still see every connect attempt.
        """
        library_logger = _LOGGER.getChild("etekcity_esf551_ble")
        if self._is_library_logging_enabled():
            library_logger.setLevel(logging.DEBUG)
            # Remove the noise filter so debug sessions see everything.
            for log_filter in list(library_logger.filters):
                if isinstance(log_filter, _NoiseFilter):
                    library_logger.removeFilter(log_filter)
        else:
            # Reset to NOTSET so the parent (integration) level applies again
            # if the user previously toggled the flag on.
            library_logger.setLevel(logging.NOTSET)
            if not any(isinstance(f, _NoiseFilter) for f in library_logger.filters):
                library_logger.addFilter(_NoiseFilter())
        return library_logger

    def _cleanup_history(self, user_profile: dict) -> None:
        """Remove invalid, old, and excess measurements from history.

        Entries with missing or unparseable timestamps or a missing/non-numeric
        weight_kg are always removed
        (corrupted data should never accumulate). The configurable retention and
        size limits are skipped entirely when the user enabled "keep history
        forever". Age-based pruning never removes a user's most recent valid
        measurement: a dormant user keeps one identity anchor so person
        detection can still match them when they return.

        Args:
            user_profile: User profile dict to cleanup.
        """
        history = user_profile.get(CONF_WEIGHT_HISTORY, [])

        if not history:
            return

        # Get configurable limits
        keep_forever = self._get_keep_history_forever()
        retention_days = self._get_history_retention_days()
        max_size = self._get_max_history_size()

        # No age cutoff when the user keeps history forever (the timestamp
        # validation below still runs unconditionally). retention_days <= 0 is
        # tolerated defensively: the raw math would make "now" the cutoff and
        # wipe the entire history.
        cutoff_time = (
            None
            if keep_forever or retention_days <= 0
            else datetime.now() - timedelta(days=retention_days)
        )
        valid_measurements = []
        newest_valid: tuple[datetime, dict] | None = None
        for m in history:
            timestamp_str = m.get("timestamp")
            if not timestamp_str:
                _LOGGER.warning("Measurement missing timestamp, removing: %s", m)
                continue
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError) as ex:
                _LOGGER.warning(
                    "Invalid timestamp format '%s' in measurement, removing: %s",
                    timestamp_str,
                    ex,
                )
                continue
            weight_kg = m.get("weight_kg")
            if isinstance(weight_kg, bool) or not isinstance(weight_kg, (int, float)):
                _LOGGER.warning(
                    "Measurement has invalid weight_kg %r, removing: %s", weight_kg, m
                )
                continue
            if newest_valid is None or parsed_timestamp >= newest_valid[0]:
                newest_valid = (parsed_timestamp, m)
            if cutoff_time is None or parsed_timestamp >= cutoff_time:
                valid_measurements.append(m)

        if not valid_measurements and newest_valid is not None:
            # Dormant user: always retain the most recent valid measurement so
            # person detection keeps an identity anchor to match against
            # (ported from multi-user-scale-core).
            valid_measurements = [newest_valid[1]]

        history[:] = valid_measurements

        # Enforce max size (keep newest); max_size <= 0 tolerated defensively
        if not keep_forever and max_size > 0 and len(history) > max_size:
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

    def check_bluetooth_available(self) -> None:
        """Check if Bluetooth is available without starting the scanner.

        This is a lightweight check that can be called before async_start()
        to verify Bluetooth availability early (e.g., before setting up platforms).

        Raises:
            BluetoothNotAvailableError: If bluetooth_manager is not available.
        """
        manager = _get_bluetooth_manager()
        if not manager:
            raise BluetoothNotAvailableError(
                "Bluetooth manager not available - Bluetooth integration may still be initializing"
            )

    def _scan_bluetooth_topology(self, manager: Any) -> _TopologyScan:
        """Discover the Bluetooth topology this integration actually uses.

        Shared by `_get_bluetooth_scanner` (to build the scanner) and
        `_topology_unchanged_since_last_start` (to relevance-filter
        scanner-registration events) so both always see the same picture.
        Scanner sources belonging to other integrations (e.g. Shelly BLE
        proxies) are deliberately invisible here - they play no part in
        this integration's scanning.
        """
        sources = manager._sources
        native = False
        # Whether BlueZ exposes org.bluez.AdvertisementMonitorManager1
        # for the adapter (bluetooth-adapters reports this as
        # `passive_scan`) - i.e. whether bleak's passive scanning can
        # work. Absent on Linux when BlueZ experimental features are
        # disabled.
        native_passive = False
        native_sources: set[str] = set()

        # Check for native adapters with better error handling
        adapter_check_ok = True
        try:
            for adapter in manager._bluetooth_adapters.adapters.values():
                if sources.get(adapter["address"]) is not None:
                    if not native:
                        native = True
                        native_passive = bool(adapter.get("passive_scan"))
                        _LOGGER.debug("Found native Bluetooth adapter: %s", adapter)
                    native_sources.add(adapter["address"])
            if not native:
                for name, details in manager._bluetooth_adapters.adapters.items():
                    if sources.get(name) is not None:
                        if not native:
                            native = True
                            native_passive = bool(details.get("passive_scan"))
                            _LOGGER.debug("Found native Bluetooth adapter: %s", details)
                        native_sources.add(name)
        except (AttributeError, KeyError) as err:
            _LOGGER.warning("Error checking native Bluetooth adapters: %s", err)
            native = False
            native_passive = False
            native_sources = set()
            adapter_check_ok = False

        # Get ESPHome proxies with error handling. Keep the whole
        # bleak-esphome ESPHomeClientData (client + cached device_info +
        # api_version), not just the APIClient: the scanners use the
        # cached info to avoid any wire round-trips at start() (a
        # device_info() call there has timed out in the field, issue #38).
        esphome_clients: list[Any] = []
        try:
            proxies = [
                item.data["source"]
                for item in self._hass.config_entries.async_entries("bluetooth")
                if item.data.get("source_domain") == "esphome"
            ]
            esphome_clients = [
                sources.get(s).connector.client.keywords["client_data"]
                for s in proxies
                if sources.get(s)
            ]
            _LOGGER.debug("Found %d ESPHome Bluetooth proxies", len(esphome_clients))
        except (AttributeError, KeyError, TypeError) as err:
            _LOGGER.warning("Error getting ESPHome clients: %s", err)
            esphome_clients = []

        return _TopologyScan(
            adapter_check_ok=adapter_check_ok,
            topology=_BluetoothTopology(
                native=native,
                native_passive=native_passive,
                native_sources=frozenset(native_sources),
                esphome_client_data=tuple(esphome_clients),
            ),
        )

    def _topology_unchanged_since_last_start(self) -> bool:
        """Whether the topology matches what the running client was built from.

        True only when a client is running (``_last_topology`` set) AND a
        fresh topology walk matches the snapshot it was built from - in
        that case a scanner-registration event carries no information for
        this entry (e.g. a Shelly proxy's scanner re-registering) and the
        restart can be skipped instead of aborting an in-flight
        measurement session. ESPHome proxies are compared by
        ``ESPHomeClientData`` *identity*: bleak-esphome builds a new
        client_data on every proxy (re)connect, so a proxy that bounced -
        even with the same source name - never compares as unchanged
        (our message callbacks died with its old connection and the
        rebuild must proceed). Any uncertainty (no snapshot, no manager,
        detection error) returns False so the restart proceeds.
        """
        last = self._last_topology
        if last is None:
            return False
        manager = _get_bluetooth_manager()
        if not manager:
            return False
        try:
            scan = self._scan_bluetooth_topology(manager)
        except Exception:  # noqa: BLE001 - unknown topology must not filter
            return False
        if not scan.adapter_check_ok:
            return False
        current = scan.topology
        return (
            current.native == last.native
            and current.native_passive == last.native_passive
            and current.native_sources == last.native_sources
            and len(current.esphome_client_data) == len(last.esphome_client_data)
            and all(
                any(client_data is held for held in last.esphome_client_data)
                for client_data in current.esphome_client_data
            )
        )

    async def _get_bluetooth_scanner(self) -> BaseBleakScanner | None:
        """Get the optimal Bluetooth scanner based on available resources.

        Returns:
            A configured Bluetooth scanner or None if no scanner could be created.

        Raises:
            BluetoothNotAvailableError: If bluetooth_manager is not available or
                no Bluetooth adapter/proxy is available.
        """
        # Reset the snapshot first: on any failure below it must not go
        # stale, or a later successful `_async_start` could promote a
        # topology that this walk didn't actually produce.
        self._pending_topology = None
        try:
            manager = _get_bluetooth_manager()
            if not manager:
                _LOGGER.debug("Bluetooth manager not available yet")
                raise BluetoothNotAvailableError(
                    "Bluetooth manager not available - Bluetooth integration may still be initializing"
                )

            scan = self._scan_bluetooth_topology(manager)
            adapter_check_ok = scan.adapter_check_ok
            native = scan.topology.native
            native_passive = scan.topology.native_passive
            esphome_clients = scan.topology.esphome_client_data

            # Snapshot the topology this scanner (or the library fallback)
            # is being built from. Promoted to `_last_topology` only once
            # the client actually starts (see `_async_start`); used by
            # `_topology_unchanged_since_last_start` to relevance-filter
            # scanner-registration events. Skipped when adapter detection
            # errored - an unknown topology must never suppress restarts.
            self._pending_topology = scan.topology if adapter_check_ok else None

            # Surface (or clear) the passive-scanning repair issue. Tied to
            # the host capability, not to which scanner path is chosen
            # below: proxies coming and going must not delete/re-create the
            # issue, since deletion erases the user's "ignore" flag and the
            # issue would resurface on every topology change. Skipped
            # entirely when adapter detection errored - a transient failure
            # must not clear (and later resurface) an ignored issue.
            if adapter_check_ok:
                from homeassistant.helpers import issue_registry as ir

                if IS_LINUX and native and not native_passive:
                    ir.async_create_issue(
                        self._hass,
                        DOMAIN,
                        PASSIVE_SCAN_ISSUE_ID,
                        is_fixable=False,
                        severity=ir.IssueSeverity.WARNING,
                        translation_key="passive_scan_unavailable",
                        learn_more_url=(
                            "https://github.com/ronnnnnnnnnnnnn/"
                            "etekcity_fitness_scale_ble#bluetooth-issues-"
                            "with-a-native-linux-adapter-bluez"
                        ),
                    )
                else:
                    ir.async_delete_issue(self._hass, DOMAIN, PASSIVE_SCAN_ISSUE_ID)

            # Initialize scanner with error handling
            scanner: BaseBleakScanner | None = None
            # Default for the library's own fallback scanner (used when we
            # return None below, i.e. native adapter with no proxies):
            # PASSIVE, which coexists with HA's shared scanner via the
            # BlueZ AdvertisementMonitor API.
            self._fallback_scanning_mode = BluetoothScanningMode.PASSIVE
            if native and not native_passive and not esphome_clients:
                # Native adapter, no proxies, but BlueZ does not expose the
                # AdvertisementMonitor API needed for passive scanning (on
                # Linux this usually means BlueZ experimental features are
                # disabled) - the library's passive scanner would fail hard
                # at start. Fall back to an active scanner instead of
                # failing setup, but explain how to get the race-free path:
                # an active scanner can race HA's shared scanner during
                # startup (org.bluez.Error.InProgress).
                self._fallback_scanning_mode = BluetoothScanningMode.ACTIVE
                if IS_LINUX:
                    _LOGGER.warning(
                        "Passive scanning is not available on this adapter "
                        "(BlueZ AdvertisementMonitor API not exposed), so "
                        "the scale library will use its own active scanner. "
                        "This can fail with org.bluez.Error.InProgress "
                        "while Home Assistant's shared scanner is starting "
                        "up. For reliable startup, enable BlueZ "
                        "experimental features (Experimental = true in "
                        "/etc/bluetooth/main.conf; requires BlueZ >= 5.56 "
                        "and kernel >= 5.10) and restart the bluetooth "
                        "service"
                    )
                else:
                    # Non-Linux: the library ignores the passive request
                    # anyway (it only applies passive on Linux), so this is
                    # just bookkeeping - no BlueZ guidance to give.
                    _LOGGER.debug(
                        "Passive scanning not available on this adapter; "
                        "the scale library will use its own active scanner"
                    )
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
        # No client is (or will shortly be) running until this attempt
        # succeeds - registration events must not be filtered while we're
        # in flux, and a failed attempt must leave every event acting as a
        # restart trigger.
        self._last_topology = None
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
            except BluetoothNotAvailableError:
                # No Bluetooth adapter or ESPHome proxy available yet
                # This is expected during startup. Re-raise so caller can decide
                # whether to schedule a retry or rely on the callback.
                _LOGGER.warning(
                    "Bluetooth not available. "
                    "Waiting for a Bluetooth adapter or ESPHome Bluetooth proxy to become available.",
                )
                raise  # Let caller handle retry logic

            # Defer the first ACTIVE-fallback start until HA has fully
            # started. An active StartDiscovery fired into the boot melee
            # (HA's shared scanner starting, adapter firmware still
            # loading) is the prime suspect for wedging some adapters into
            # a persistent `org.bluez.Error.InProgress`, so wait out the
            # window instead of racing it. The listener routes through
            # `_async_registration_changed`, so the normal lock, debounce
            # and backoff machinery applies when it fires. Entities stay
            # idle until then; this branch is unreachable for reloads and
            # late setups (state is already `running` then). The state
            # check matches HA's own `async_at_started` helper - NOT
            # `is_running`, which is already True during CoreState.starting
            # (the <=15s wrap-up window between START and STARTED, i.e.
            # peak boot churn - exactly what we're deferring past).
            if (
                scanner is None
                and self._fallback_scanning_mode == BluetoothScanningMode.ACTIVE
                and self._hass.state is not CoreState.running
            ):
                if self._started_listener_unsub is None:
                    self._started_listener_unsub = self._hass.bus.async_listen_once(
                        EVENT_HOMEASSISTANT_STARTED, self._on_ha_started
                    )
                    _LOGGER.info(
                        "Deferring active-mode scanner start for %s until "
                        "Home Assistant has finished starting, to avoid "
                        "the BlueZ startup race",
                        self.address,
                    )
                return

            # A start is actually proceeding (a scanner became available,
            # the passive fallback applies, or HA finished starting). A
            # still-pending deferred start from an earlier pass would only
            # force a redundant rebuild when STARTED fires - cancel it.
            # Failure recovery isn't lost: if this start throws, the
            # backoff machinery schedules the retry, not the listener.
            if self._started_listener_unsub is not None:
                self._started_listener_unsub()
                self._started_listener_unsub = None

            # Initialize client based on scale model
            try:
                library_logger = self._configure_library_logger()

                client_cls = SCALE_CLASSES.get(self._scale_model)
                if client_cls is None:
                    # Unknown or legacy model value: fall back to the ESF-551 client.
                    _LOGGER.warning(
                        "Unknown scale model %r; using the ESF-551 client",
                        self._scale_model,
                    )
                    client_cls = SCALE_CLASSES[ScaleModel.ESF551]
                _LOGGER.debug(
                    "Initializing new %s client (scale_model=%s)",
                    client_cls.__name__,
                    self._scale_model,
                )
                # NOTE: pass everything beyond (address, callback, display
                # unit) by keyword — the concrete classes disagree on the
                # positional order after those (FIT8SScale takes logger where
                # the GATT models take cooldown_seconds). Do not pass
                # cooldown_seconds at all: each class sets its own
                # hardware-appropriate default (e.g. FIT-8S uses a 10s window
                # to deduplicate its advertising bursts; overriding it with 0
                # would deliver duplicate measurements per weigh-in).
                # `scanning_mode` only matters when `scanner` is None (the
                # library then builds its own fallback scanner with it);
                # `_get_bluetooth_scanner` downgrades it from PASSIVE to
                # ACTIVE when the native adapter can't do passive scanning.
                self._client = client_cls(
                    self.address,
                    self.update_listeners,
                    self._display_unit,
                    scanning_mode=self._fallback_scanning_mode,
                    bleak_scanner_backend=scanner,
                    logger=library_logger,
                )

                await asyncio.wait_for(self._client.async_start(), timeout=30.0)
                _LOGGER.debug("Scale client started successfully")
                # The client is running on the topology `_get_bluetooth_scanner`
                # walked (NOT one recomputed now: a proxy that bounced during
                # the slow BLE start above must still register as changed, or
                # the queued registration event would be filtered out and the
                # scanner left listening on a dead connection). From here on,
                # registration events that don't change this snapshot are
                # irrelevant and get skipped instead of aborting the client.
                self._last_topology = self._pending_topology
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
        self._hass.async_create_task(self._async_registration_changed(registration))

    async def _async_registration_changed(
        self, registration: HaScannerRegistration | None = None
    ) -> None:
        """Restart the scale client to pick up a new/changed scanner.

        Guarded by an exponential-backoff circuit breaker. Rebuilding the
        whole BLE pipeline on every scanner-registration event with no
        backoff means that once a restart starts failing persistently
        (e.g. host BlueZ without passive-scanning support), each ESPHome
        proxy reconnect triggers another full failing rebuild — and via the
        leak in ``BleakScannerHybrid.start()`` fixed above, each one adds
        more permanently-subscribed scanners on top, saturating the HA
        event loop for hours. The backoff now governs only the
        *self-scheduled* retries (the timer fired by
        ``_schedule_restart_retry``, delay growing exponentially, 30s ->
        30min cap): a *real* registration event (a scanner actually being
        added/removed) instead cancels any pending backoff retry and
        restarts immediately, since the event itself could be exactly
        what fixes the failure (e.g. an ESPHome proxy reconnecting).

        Registration events aren't rate-limited by HA itself, though, so a
        flapping/bootlooping proxy could otherwise drive restart attempts
        faster than the backoff is meant to allow (no leak, but repeated
        real I/O load). `REGISTRATION_PREEMPT_DEBOUNCE_SECONDS` is a hard
        floor on *every* restart attempt made from here, measured from
        the end of the previous attempt (so even a slow attempt is
        followed by a full quiet window), regardless of
        whether a backoff retry happens to be pending - covering both a
        flapping proxy during an active backoff and one whose restarts
        happen to keep succeeding (no backoff pending at all). The check
        and the restart attempt both happen under `_lock`, so a burst of
        events arriving while a restart is already in flight queue up on
        the lock and then re-check the floor against the latest attempt
        (including one that just finished) instead of all slipping through
        on a stale check made before they were queued. Events landing
        inside the floor are deferred rather than dropped: they coalesce
        into a single retry scheduled for the floor's expiry (pulling in
        any pending backoff retry, whose delay is always longer than the
        floor), preserving the invariant that every registration event
        leads to a restart attempt - either immediately or as soon as the
        floor allows. A successful restart resets the backoff.
        """
        async with self._lock:
            if self._stopped:
                # A registration event scheduled this call before
                # `async_stop` ran (e.g. right before unload) but only
                # got the lock afterwards - `async_stop` already tore
                # down `self._client`; don't build a new one.
                _LOGGER.debug(
                    "Registration-change event arrived after the "
                    "coordinator was stopped; ignoring"
                )
                return
            if self._topology_unchanged_since_last_start():
                # The event is real but irrelevant to this entry: the
                # native adapter and every ESPHome proxy we're built on are
                # exactly as the running client left them. Typically another
                # integration's scanner re-registering (e.g. Shelly BLE
                # proxies bounce on every WebSocket reconnect) - restarting
                # here would abort an in-flight connection/measurement for
                # nothing. Deferred debounce retries also land here and
                # become no-ops when an interim rebuild already captured
                # the current topology.
                source = getattr(getattr(registration, "scanner", None), "source", None)
                _LOGGER.debug(
                    "Scanner registration changed (source: %s) but the "
                    "Bluetooth topology this entry uses is unchanged; "
                    "skipping scale client restart",
                    source or "unknown",
                )
                return
            since_last_attempt = (
                None
                if self._last_restart_attempt_monotonic is None
                else time.monotonic() - self._last_restart_attempt_monotonic
            )
            if (
                since_last_attempt is not None
                and since_last_attempt < REGISTRATION_PREEMPT_DEBOUNCE_SECONDS
            ):
                # Defer, don't drop: restarts are edge-triggered (each one
                # reads live scanner state only at the moment it runs), so
                # an event discarded here would be lost until an unrelated
                # future event fired - e.g. a second proxy registering
                # seconds after the first would never get picked up.
                # Coalesce it into a single retry at the floor's expiry
                # instead. Any pending backoff retry is pulled in to that
                # same expiry (`_schedule_restart_retry` replaces it): its
                # delay is always longer than the floor (30s base vs 5s),
                # and per the pre-emption rule below a real event should
                # be acted on as soon as the floor allows.
                remaining = REGISTRATION_PREEMPT_DEBOUNCE_SECONDS - since_last_attempt
                self._schedule_restart_retry(remaining)
                _LOGGER.debug(
                    "Registration-change event arrived %.1fs after the "
                    "last restart attempt (< %ds debounce floor); "
                    "deferring restart by %.1fs",
                    since_last_attempt,
                    REGISTRATION_PREEMPT_DEBOUNCE_SECONDS,
                    remaining,
                )
                return
            if self._restart_retry_unsub is not None:
                # A backoff retry is already scheduled, but this is a real
                # registration-change event, not the retry timer firing
                # (the timer's own callback clears `_restart_retry_unsub`
                # before calling back in here) - cancel the pending retry
                # and try now, since the event itself could be exactly
                # what fixes the failure.
                _LOGGER.debug(
                    "Scale client restart already scheduled (backoff after "
                    "%d failure(s)); registration-change event pre-empting "
                    "it for an immediate retry",
                    self._restart_failures,
                )
                self._restart_retry_unsub()
                self._restart_retry_unsub = None
            _LOGGER.debug("BT scanner registration changed; restarting scale client")
            try:
                await self._async_start()
            except Exception:
                self._restart_failures += 1
                delay = min(
                    RESTART_BACKOFF_BASE_SECONDS * (2 ** (self._restart_failures - 1)),
                    RESTART_BACKOFF_MAX_SECONDS,
                )
                _LOGGER.exception(
                    "Failed to restart scale client after scanner change "
                    "(consecutive failure #%d); next retry in %ds",
                    self._restart_failures,
                    delay,
                )
                self._schedule_restart_retry(delay)
            else:
                if self._restart_failures:
                    _LOGGER.info(
                        "Scale client restart succeeded after %d failed "
                        "attempt(s); resetting backoff",
                        self._restart_failures,
                    )
                self._restart_failures = 0
            finally:
                # Stamp the *end* of the attempt, not the start: the floor
                # guarantees a quiet window between the end of one attempt
                # and the start of the next. Stamped at the start, events
                # queued on the lock behind a slow attempt (scanner startup
                # is real BLE I/O, up to 30s) would sail through the floor
                # the moment the lock releases - back-to-back rebuilds
                # exactly when rebuilds are most expensive.
                self._last_restart_attempt_monotonic = time.monotonic()

    def _schedule_restart_retry(self, delay: float) -> None:
        """Schedule a single delayed restart retry.

        Serves both the exponential-backoff retries after a failed restart
        and the deferred restarts coalesced by the debounce floor. Cancels
        any retry already pending before scheduling, so only one retry is
        ever pending at a time - a new registration event pre-empts it
        (immediately when outside the debounce floor, rescheduled to the
        floor's expiry when inside it) rather than a second one being
        scheduled alongside it. The handle is also cleared when the timer
        fires and in `async_stop`.
        """
        if self._restart_retry_unsub is not None:
            self._restart_retry_unsub()
            self._restart_retry_unsub = None

        @callback
        def _retry(_now) -> None:
            self._restart_retry_unsub = None
            self._hass.async_create_task(self._async_registration_changed())

        self._restart_retry_unsub = async_call_later(self._hass, delay, _retry)

    @callback
    def _on_ha_started(self, _event) -> None:
        """Run the deferred first start once HA has fully started.

        Routed through `_async_registration_changed` rather than calling
        `_async_start` directly, so the deferred start gets the same
        `_stopped` check, debounce and backoff treatment as any other
        restart trigger.
        """
        self._started_listener_unsub = None
        self._hass.async_create_task(self._async_registration_changed())

    async def async_start(self) -> None:
        """Start the coordinator and initialize the scale client.

        This method sets up the EtekcitySmartFitnessScale client and starts
        listening for updates from the scale.

        Raises:
            BluetoothNotAvailableError: If Bluetooth manager or adapters are not
                available. Caller should handle this by raising ConfigEntryNotReady.
        """
        _LOGGER.debug(
            "Starting ScaleDataUpdateCoordinator for address: %s", self.address
        )

        # Clean up any existing registration callback
        if self._scanner_change_cb_unregister:
            self._scanner_change_cb_unregister()
            self._scanner_change_cb_unregister = None

        # Register for scanner changes (if bluetooth_manager is available)
        # This callback will restart the client when adapters/proxies change
        bluetooth_manager = _get_bluetooth_manager()
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
            except BluetoothNotAvailableError:
                # Let the error bubble up - caller will raise ConfigEntryNotReady
                # which triggers Home Assistant's automatic retry with backoff
                raise
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

    async def async_stop(self) -> None:
        """Stop the coordinator and clean up resources.

        Runs under `_lock` so it can't race an in-flight `_async_start()`
        (from `async_start` or `_async_registration_changed`) - without
        it, a restart racing with unload could leave a freshly-built
        client running after this returns. `_stopped` is set first, before
        touching `self._client`, so a registration event that was already
        scheduled before this call notices once it's its turn at the lock
        and bails out instead of building a new client.
        """
        _LOGGER.debug(
            "Stopping ScaleDataUpdateCoordinator for address: %s", self.address
        )
        async with self._lock:
            self._stopped = True
            # Cancel any pending backoff/deferred retry so a stopped or
            # unloaded coordinator can't restart itself later. Done under
            # the lock, right after setting `_stopped`: an in-flight
            # restart holds the lock for its whole attempt, including
            # scheduling a new retry on failure, so by the time we get the
            # lock ourselves any such retry has already been scheduled and
            # is visible here to cancel - it can't be scheduled afterwards
            # and slip past this check, since `_stopped` makes any later
            # restart attempt a no-op before it would reach that point.
            if self._restart_retry_unsub is not None:
                self._restart_retry_unsub()
                self._restart_retry_unsub = None
            # Cancel a pending deferred first start (HA-started listener)
            # for the same reason - a stopped coordinator must not revive.
            if self._started_listener_unsub is not None:
                self._started_listener_unsub()
                self._started_listener_unsub = None
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

            # No running client - release the topology snapshot (and the
            # ESPHomeClientData references it holds alive)
            self._last_topology = None
            self._pending_topology = None

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
                    notify_domain, notify_name = parse_notify_service(service_name)
                    await self._hass.services.async_call(
                        notify_domain,
                        notify_name,
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
            Dictionary with only raw measurements (weight, impedance, heart_rate).
        """
        raw_measurements = {}
        if "weight" in data.measurements:
            raw_measurements["weight"] = data.measurements["weight"]
        if "impedance" in data.measurements:
            raw_measurements["impedance"] = data.measurements["impedance"]
        if "heart_rate" in data.measurements:
            raw_measurements["heart_rate"] = data.measurements["heart_rate"]
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
                        notify_domain, notify_name = parse_notify_service(service)
                        await self._hass.services.async_call(
                            notify_domain,
                            notify_name,
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
                from etekcity_esf551_ble import BodyMetrics, Sex, calc_age
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

                            sex_str = user_profile.get("sex", "male")
                            sex = (
                                Sex.Female
                                if (sex_str or "").lower() == "female"
                                else Sex.Male
                            )
                            age = calc_age(birthdate)
                            body_metrics = BodyMetrics(
                                weight_kg,
                                height_m,
                                age,
                                sex,
                                impedance,
                                athlete=bool(user_profile.get(CONF_ATHLETE, False)),
                            )
                            metrics_dict = body_metrics.as_dict()

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

        # Format time using display preferences (country heuristic, Babel fallback)
        time_display = self._format_notification_time(timestamp)

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

                notify_domain, notify_name = parse_notify_service(service_name)
                await self._hass.services.async_call(
                    notify_domain,
                    notify_name,
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

            # User has usable history - get last measurement for ranking.
            # Use the filtered accessor: entries with invalid weight_kg are
            # skipped, so the history can come back empty here.
            weight_history = self.get_user_history(user_id)
            if not weight_history:
                other_users.append((user_id, user_name))
                continue
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

        timestamp_display = self._format_notification_timestamp(timestamp)
        message = (
            f"**Scale: {device_name}**\n\n"
            f"**Multiple users could match this measurement**\n\n"
            f"{measurement_info}\n"
            f"Timestamp: `{timestamp_display}`\n\n"
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
                from etekcity_esf551_ble import BodyMetrics, Sex, calc_age
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

                            sex_str = user_profile.get("sex", "male")
                            sex = (
                                Sex.Female
                                if (sex_str or "").lower() == "female"
                                else Sex.Male
                            )
                            age = calc_age(birthdate)
                            body_metrics = BodyMetrics(
                                weight_kg,
                                height_m,
                                age,
                                sex,
                                impedance,
                                athlete=bool(user_profile.get(CONF_ATHLETE, False)),
                            )
                            metrics_dict = body_metrics.as_dict()

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
            notify_domain, notify_name = parse_notify_service(service_name)
            self._hass.async_create_task(
                self._hass.services.async_call(
                    notify_domain,
                    notify_name,
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

    def ignore_candidate_for_pending_measurement(
        self, timestamp: str, user_id: str
    ) -> bool:
        """Remove a user from candidates for a pending measurement.

        When "Not Me" is pressed on a mobile notification, this updates the
        notification by removing that user from the candidate list. If no
        candidates remain, the notification is dismissed.

        Args:
            timestamp: ISO timestamp of the pending measurement.
            user_id: The user ID to remove from candidates.

        Returns:
            True if the candidate was removed and notification was updated.
        """
        pending_data = self._pending_measurements.get(timestamp)
        if pending_data is None:
            return False

        candidates = pending_data["candidates"]
        updated_candidates = [c for c in candidates if c != user_id]
        if len(updated_candidates) == len(candidates):
            return False

        if not updated_candidates:
            # No candidates left - dismiss notification and remove pending
            del self._pending_measurements[timestamp]
            self._ambiguous_notifications.discard(timestamp)
            notification_id = f"etekcity_scale_{self.address}_{timestamp}"
            persistent_notification.dismiss(
                self._hass,
                notification_id=notification_id,
            )
            tag = f"scale_measurement_{timestamp}"
            for _uid, service_name in pending_data.get("notified_mobile_services", []):
                notify_domain, notify_name = parse_notify_service(service_name)
                self._hass.async_create_task(
                    self._hass.services.async_call(
                        notify_domain,
                        notify_name,
                        {"message": "clear_notification", "data": {"tag": tag}},
                    )
                )
            self._notify_diagnostic_sensors()
            return True

        # Update candidates and re-send notification
        pending_data["candidates"] = updated_candidates
        old_notified = list(pending_data.get("notified_mobile_services", []))
        pending_data["notified_mobile_services"] = []

        # Clear old mobile notifications
        tag = f"scale_measurement_{timestamp}"
        for _uid, service_name in old_notified:
            notify_domain, notify_name = parse_notify_service(service_name)
            self._hass.async_create_task(
                self._hass.services.async_call(
                    notify_domain,
                    notify_name,
                    {"message": "clear_notification", "data": {"tag": tag}},
                )
            )

        # Dismiss persistent notification
        notification_id = f"etekcity_scale_{self.address}_{timestamp}"
        persistent_notification.dismiss(
            self._hass,
            notification_id=notification_id,
        )

        # Re-create notification with updated candidates
        measurements = pending_data["measurements"]
        weight_kg = measurements.get("weight")
        impedance = measurements.get("impedance")

        async def _recreate_notification() -> None:
            try:
                await self._create_ambiguous_notification(
                    weight_kg,
                    impedance,
                    updated_candidates,
                    timestamp,
                )
            except Exception as ex:
                _LOGGER.error(
                    "Failed to re-create notification after ignore (timestamp: %s): %s",
                    timestamp,
                    ex,
                )

        self._hass.async_create_task(_recreate_notification())
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
