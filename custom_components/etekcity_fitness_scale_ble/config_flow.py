"""Config flow for etekcity_fitness_scale_ble integration."""

from __future__ import annotations

import dataclasses
from datetime import datetime
import logging
import re
import unicodedata
from typing import Any

import voluptuous as vol

from homeassistant.components.bluetooth import (
    BluetoothServiceInfo,
    async_discovered_service_info,
)
from homeassistant.config_entries import ConfigEntry, ConfigFlow, OptionsFlow
from homeassistant.const import CONF_ADDRESS, CONF_UNIT_SYSTEM, UnitOfLength, UnitOfMass
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers import entity_registry as er
from homeassistant.util.unit_conversion import DistanceConverter

from .const import (
    CONF_BIRTHDATE,
    CONF_BODY_METRICS_ENABLED,
    CONF_CALC_BODY_METRICS,
    CONF_CREATED_AT,
    CONF_ENABLE_LIBRARY_LOGGING,
    CONF_FEET,
    CONF_HEIGHT,
    CONF_HISTORY_RETENTION_DAYS,
    CONF_INCHES,
    CONF_MAX_HISTORY_SIZE,
    CONF_MOBILE_NOTIFY_SERVICES,
    CONF_PERSON_ENTITY,
    CONF_SCALE_DISPLAY_UNIT,
    CONF_SEX,
    CONF_UPDATED_AT,
    CONF_USER_ID,
    CONF_USER_NAME,
    CONF_USER_PROFILES,
    DOMAIN,
    HISTORY_RETENTION_DAYS,
    MAX_HISTORY_SIZE,
    get_sensor_unique_id,
)
from .sensor import SENSOR_DESCRIPTIONS

_LOGGER = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]")


def _get_mobile_notify_services(hass) -> dict[str, str]:
    """Discover available mobile_app notify services.

    Returns:
        Dict mapping service_name -> display_name (e.g., "mobile_app_johns_iphone" -> "Johns iPhone")
    """
    services = {}
    notify_services = hass.services.async_services().get("notify", {})

    for service_name in notify_services.keys():
        if service_name.startswith("mobile_app_"):
            # Format: "mobile_app_johns_iphone" -> "Johns iPhone"
            display_name = (
                service_name.replace("mobile_app_", "").replace("_", " ").title()
            )
            services[service_name] = display_name

    return services


def _create_user_id(display_name: str, existing_profiles: list[dict]) -> str:
    """Create a unique, human-readable user ID from a display name."""
    base = (
        _SLUG_RE.sub(
            "",
            unicodedata.normalize("NFKD", display_name)
            .encode("ascii", "ignore")
            .decode()
            .lower(),
        )
        or "user"
    )

    existing_ids = {p[CONF_USER_ID] for p in existing_profiles}

    slug = base
    idx = 2
    while slug in existing_ids:
        slug = f"{base}{idx}"
        idx += 1

    return slug


def _validate_user_id_unique(
    user_id: str,
    user_profiles: list[dict],
    exclude_user_id: str | None = None,
) -> bool:
    """Validate that user_id is unique among profiles.

    Args:
        user_id: The user ID to check.
        user_profiles: List of existing user profiles.
        exclude_user_id: Optional user_id to exclude from check (for edits).

    Returns:
        True if user_id is unique, False otherwise.
    """
    for profile in user_profiles:
        profile_id = profile.get(CONF_USER_ID)
        if profile_id == user_id and profile_id != exclude_user_id:
            return False
    return True


def _validate_user_name_not_empty(user_name: str) -> bool:
    """Validate that user_name is not empty.

    Args:
        user_name: The user name to check.

    Returns:
        True if user_name is not empty, False otherwise.
    """
    return bool(user_name and user_name.strip())


def _validate_user_id_not_reserved(user_id: str) -> bool:
    """Validate that user_id is not a reserved value.

    The empty string "" is reserved for v1 compatibility.

    Args:
        user_id: The user ID to check.

    Returns:
        True if user_id is not reserved, False otherwise.
    """
    return user_id != ""


def _validate_person_entity_unique(
    person_entity: str | None,
    user_profiles: list[dict],
    exclude_user_id: str | None = None,
) -> bool:
    """Validate that person_entity is not already used by another user.

    Args:
        person_entity: The person entity to check (None/empty is okay).
        user_profiles: List of existing user profiles.
        exclude_user_id: Optional user_id to exclude from check (for edits).

    Returns:
        True if person_entity is unique or None/empty, False otherwise.
    """
    # None or empty person entity is always valid (user may not want to link)
    if not person_entity:
        return True

    for profile in user_profiles:
        profile_person = profile.get(CONF_PERSON_ENTITY)
        profile_id = profile.get(CONF_USER_ID)

        # Skip if this is the user being edited
        if profile_id == exclude_user_id:
            continue

        # Check for duplicate person entity
        if profile_person == person_entity:
            return False

    return True


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

    VERSION = 2
    _entry: ConfigEntry

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._discovered_devices: dict[str, Discovery] = {}
        self._initialize_schema_dicts()

    def _initialize_schema_dicts(self) -> None:
        """Initialize metric and imperial schema dictionaries."""
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

        self.context["title_placeholders"] = {"name": title(discovery_info)}

        return await self.async_step_bluetooth_confirm()

    async def async_step_bluetooth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm discovery."""
        if user_input is not None:
            # Store display unit in context
            self.context[CONF_SCALE_DISPLAY_UNIT] = user_input[CONF_SCALE_DISPLAY_UNIT]

            # Always proceed to create first user (required)
            return await self.async_step_add_first_user()

        return self.async_show_form(
            step_id="bluetooth_confirm",
            description_placeholders=self.context["title_placeholders"],
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_SCALE_DISPLAY_UNIT, default=UnitOfMass.KILOGRAMS
                    ): vol.In(
                        {
                            UnitOfMass.KILOGRAMS: "Metric (kg)",
                            UnitOfMass.POUNDS: "Imperial (lbs)",
                        }
                    ),
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

    async def async_step_add_first_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add first user during initial setup."""
        if user_input is not None:
            # Check if body metrics is enabled
            if user_input.get(CONF_BODY_METRICS_ENABLED, False):
                # Store basic user info in context and proceed to body metrics
                self.context[CONF_USER_NAME] = user_input[CONF_USER_NAME]
                self.context[CONF_PERSON_ENTITY] = user_input.get(CONF_PERSON_ENTITY)
                self.context[CONF_MOBILE_NOTIFY_SERVICES] = user_input.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                )
                return await self.async_step_add_first_user_body_metrics()

            # No body metrics - create entry with basic user profile
            user_profile = {
                CONF_USER_ID: _create_user_id(user_input[CONF_USER_NAME], []),
                CONF_USER_NAME: user_input[CONF_USER_NAME],
                CONF_PERSON_ENTITY: user_input.get(CONF_PERSON_ENTITY),
                CONF_MOBILE_NOTIFY_SERVICES: user_input.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                ),
                CONF_BODY_METRICS_ENABLED: False,
                CONF_CREATED_AT: datetime.now().isoformat(),
                CONF_UPDATED_AT: datetime.now().isoformat(),
            }

            return self.async_create_entry(
                title=self.context["title_placeholders"]["name"],
                data={
                    CONF_SCALE_DISPLAY_UNIT: self.context[CONF_SCALE_DISPLAY_UNIT],
                    CONF_USER_PROFILES: [user_profile],
                },
            )

        # Build schema
        schema = {
            vol.Required(CONF_USER_NAME, default="Me"): str,
        }

        # Match the options flow: use an entity selector for person
        schema[vol.Optional(CONF_PERSON_ENTITY)] = selector.EntitySelector(
            selector.EntitySelectorConfig(domain="person")
        )

        # Add mobile notify services selector if any are available
        available_mobile_services = _get_mobile_notify_services(self.hass)
        if available_mobile_services:
            schema[vol.Optional(CONF_MOBILE_NOTIFY_SERVICES)] = cv.multi_select(
                available_mobile_services
            )

        schema[vol.Required(CONF_BODY_METRICS_ENABLED, default=False)] = cv.boolean

        return self.async_show_form(
            step_id="add_first_user",
            data_schema=vol.Schema(schema),
        )

    async def async_step_add_first_user_body_metrics(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect body metrics for first user."""
        if user_input is not None:
            # Convert height to cm based on display unit
            display_unit = self.context[CONF_SCALE_DISPLAY_UNIT]

            if display_unit == UnitOfMass.KILOGRAMS:
                height_cm = user_input[CONF_HEIGHT]
            else:
                # Imperial - convert to cm
                height_cm = round(
                    DistanceConverter.convert(
                        user_input[CONF_FEET],
                        UnitOfLength.FEET,
                        UnitOfLength.CENTIMETERS,
                    )
                    + DistanceConverter.convert(
                        user_input[CONF_INCHES],
                        UnitOfLength.INCHES,
                        UnitOfLength.CENTIMETERS,
                    )
                )

            # Create user profile with body metrics
            user_profile = {
                CONF_USER_ID: _create_user_id(self.context[CONF_USER_NAME], []),
                CONF_USER_NAME: self.context[CONF_USER_NAME],
                CONF_PERSON_ENTITY: self.context.get(CONF_PERSON_ENTITY),
                CONF_MOBILE_NOTIFY_SERVICES: self.context.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                ),
                CONF_BODY_METRICS_ENABLED: True,
                CONF_SEX: user_input[CONF_SEX],
                CONF_BIRTHDATE: user_input[CONF_BIRTHDATE],
                CONF_HEIGHT: height_cm,
                CONF_CREATED_AT: datetime.now().isoformat(),
                CONF_UPDATED_AT: datetime.now().isoformat(),
            }

            return self.async_create_entry(
                title=self.context["title_placeholders"]["name"],
                data={
                    CONF_SCALE_DISPLAY_UNIT: self.context[CONF_SCALE_DISPLAY_UNIT],
                    CONF_USER_PROFILES: [user_profile],
                },
            )

        # Build schema based on display unit
        display_unit = self.context[CONF_SCALE_DISPLAY_UNIT]

        schema = {
            vol.Required(CONF_SEX): vol.In(["Male", "Female"]),
            vol.Required(CONF_BIRTHDATE): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.DATE,
                ),
            ),
        }

        if display_unit == UnitOfMass.KILOGRAMS:
            schema[vol.Required(CONF_HEIGHT, default=170)] = vol.All(
                vol.Coerce(int), vol.Range(min=100, max=250)
            )
        else:
            schema[vol.Required(CONF_FEET, default=5)] = vol.All(
                vol.Coerce(int), vol.Range(min=3, max=8)
            )
            schema[vol.Required(CONF_INCHES, default=7)] = vol.All(
                vol.Coerce(float), vol.Range(min=0, max=11.5)
            )

        return self.async_show_form(
            step_id="add_first_user_body_metrics",
            data_schema=vol.Schema(schema),
        )

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

            # Store display unit and title in context
            self.context[CONF_SCALE_DISPLAY_UNIT] = user_input[CONF_SCALE_DISPLAY_UNIT]
            self.context["title_placeholders"] = {"name": discovery.title}

            # Always proceed to create first user (required)
            return await self.async_step_add_first_user()

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
                    vol.Required(
                        CONF_SCALE_DISPLAY_UNIT, default=UnitOfMass.KILOGRAMS
                    ): vol.In(
                        {
                            UnitOfMass.KILOGRAMS: "Metric (kg)",
                            UnitOfMass.POUNDS: "Imperial (lbs)",
                        }
                    ),
                }
            ),
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Get the options flow for this handler."""
        return ScaleOptionsFlow(config_entry)


class ScaleOptionsFlow(OptionsFlow):
    """Handle options flow for the scale integration."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow.

        Note: self.config_entry is automatically provided by OptionsFlow base class.
        Do not assign it explicitly as it's deprecated in HA 2025.12+.
        """
        # Initialize from config_entry data (config_entry parameter is still needed for init)
        self.user_profiles = config_entry.data.get(CONF_USER_PROFILES, [])
        self.display_unit = config_entry.data.get(
            CONF_SCALE_DISPLAY_UNIT, UnitOfMass.KILOGRAMS
        )
        self.history_retention_days = config_entry.data.get(
            CONF_HISTORY_RETENTION_DAYS, HISTORY_RETENTION_DAYS
        )
        self.max_history_size = config_entry.data.get(
            CONF_MAX_HISTORY_SIZE, MAX_HISTORY_SIZE
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        # Build menu options
        menu_options = ["add_user", "scale_settings", "advanced_settings"]
        if self.user_profiles:
            menu_options.insert(1, "edit_user")
            menu_options.insert(2, "remove_user")

        return self.async_show_menu(
            step_id="init",
            menu_options=menu_options,
        )

    async def async_step_add_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add a new user."""
        if user_input is not None:
            # Validate user_name is not empty
            user_name = user_input[CONF_USER_NAME]
            person_entity = user_input.get(CONF_PERSON_ENTITY)

            errors = {}

            if not _validate_user_name_not_empty(user_name):
                errors["base"] = "empty_user_name"

            # Validate person entity is unique
            if person_entity and not _validate_person_entity_unique(
                person_entity, self.user_profiles
            ):
                errors["person_entity"] = "duplicate_person_entity"

            if errors:
                # Re-show form with errors (ordered: name → person link → mobile devices → features)
                available_mobile_services = _get_mobile_notify_services(self.hass)

                schema = {
                    vol.Required(CONF_USER_NAME): str,
                    vol.Optional(CONF_PERSON_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(domain="person")
                    ),
                }

                # Add mobile notify services selector if any are available
                if available_mobile_services:
                    schema[vol.Optional(CONF_MOBILE_NOTIFY_SERVICES)] = cv.multi_select(
                        available_mobile_services
                    )

                schema[vol.Required(CONF_BODY_METRICS_ENABLED, default=False)] = (
                    cv.boolean
                )

                return self.async_show_form(
                    step_id="add_user",
                    data_schema=vol.Schema(schema),
                    errors=errors,
                )

            # Check if body metrics is enabled
            if user_input.get(CONF_BODY_METRICS_ENABLED, False):
                # Store basic user info in options context and proceed to body metrics
                self.context[CONF_USER_NAME] = user_name
                self.context[CONF_PERSON_ENTITY] = user_input.get(CONF_PERSON_ENTITY)
                self.context[CONF_MOBILE_NOTIFY_SERVICES] = user_input.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                )
                return await self.async_step_add_user_body_metrics()

            # Generate new user_id
            new_user_id = _create_user_id(user_name, self.user_profiles)

            # Validate uniqueness (defensive, should always pass with slug logic)
            if not _validate_user_id_unique(new_user_id, self.user_profiles):
                _LOGGER.error("Generated duplicate user_id: %s", new_user_id)
                return self.async_abort(reason="user_id_generation_failed")

            # No body metrics - create basic user profile
            user_profile = {
                CONF_USER_ID: new_user_id,
                CONF_USER_NAME: user_name,
                CONF_PERSON_ENTITY: user_input.get(CONF_PERSON_ENTITY),
                CONF_MOBILE_NOTIFY_SERVICES: user_input.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                ),
                CONF_BODY_METRICS_ENABLED: False,
                CONF_CREATED_AT: datetime.now().isoformat(),
                CONF_UPDATED_AT: datetime.now().isoformat(),
            }

            # Add user to profiles
            updated_profiles = self.user_profiles + [user_profile]

            # Update config entry
            new_data = {**self.config_entry.data, CONF_USER_PROFILES: updated_profiles}
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )

            # Reload the integration
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)

            return self.async_create_entry(title="", data={})

        # Build schema (ordered: name → person link → mobile devices → features)
        available_mobile_services = _get_mobile_notify_services(self.hass)

        schema = {
            vol.Required(CONF_USER_NAME): str,
            vol.Optional(CONF_PERSON_ENTITY): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="person")
            ),
        }

        # Add mobile notify services selector if any are available
        if available_mobile_services:
            schema[vol.Optional(CONF_MOBILE_NOTIFY_SERVICES)] = cv.multi_select(
                available_mobile_services
            )

        schema[vol.Required(CONF_BODY_METRICS_ENABLED, default=False)] = cv.boolean

        return self.async_show_form(
            step_id="add_user",
            data_schema=vol.Schema(schema),
        )

    async def async_step_add_user_body_metrics(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect body metrics for new user."""
        if user_input is not None:
            # Validate user_name from context is not empty
            user_name = self.context[CONF_USER_NAME]
            if not _validate_user_name_not_empty(user_name):
                _LOGGER.error(
                    "Empty user_name in context during body metrics collection"
                )
                return self.async_abort(reason="invalid_user_name")

            # Convert height to cm based on display unit
            if self.display_unit == UnitOfMass.KILOGRAMS:
                height_cm = user_input[CONF_HEIGHT]
            else:
                # Imperial - convert to cm
                height_cm = round(
                    DistanceConverter.convert(
                        user_input[CONF_FEET],
                        UnitOfLength.FEET,
                        UnitOfLength.CENTIMETERS,
                    )
                    + DistanceConverter.convert(
                        user_input[CONF_INCHES],
                        UnitOfLength.INCHES,
                        UnitOfLength.CENTIMETERS,
                    )
                )

            # Generate new user_id
            new_user_id = _create_user_id(user_name, self.user_profiles)

            # Validate uniqueness (defensive, should always pass with slug logic)
            if not _validate_user_id_unique(new_user_id, self.user_profiles):
                _LOGGER.error("Generated duplicate user_id: %s", new_user_id)
                return self.async_abort(reason="user_id_generation_failed")

            # Create user profile with body metrics
            user_profile = {
                CONF_USER_ID: new_user_id,
                CONF_USER_NAME: user_name,
                CONF_PERSON_ENTITY: self.context.get(CONF_PERSON_ENTITY),
                CONF_MOBILE_NOTIFY_SERVICES: self.context.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                ),
                CONF_BODY_METRICS_ENABLED: True,
                CONF_SEX: user_input[CONF_SEX],
                CONF_BIRTHDATE: user_input[CONF_BIRTHDATE],
                CONF_HEIGHT: height_cm,
                CONF_CREATED_AT: datetime.now().isoformat(),
                CONF_UPDATED_AT: datetime.now().isoformat(),
            }

            # Add user to profiles
            updated_profiles = self.user_profiles + [user_profile]

            # Update config entry
            new_data = {**self.config_entry.data, CONF_USER_PROFILES: updated_profiles}
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )

            # Reload the integration
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)

            return self.async_create_entry(title="", data={})

        # Build schema based on display unit
        schema = {
            vol.Required(CONF_SEX): vol.In(["Male", "Female"]),
            vol.Required(CONF_BIRTHDATE): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.DATE,
                ),
            ),
        }

        if self.display_unit == UnitOfMass.KILOGRAMS:
            schema[vol.Required(CONF_HEIGHT, default=170)] = vol.All(
                vol.Coerce(int), vol.Range(min=100, max=250)
            )
        else:
            schema[vol.Required(CONF_FEET, default=5)] = vol.All(
                vol.Coerce(int), vol.Range(min=3, max=8)
            )
            schema[vol.Required(CONF_INCHES, default=7)] = vol.All(
                vol.Coerce(float), vol.Range(min=0, max=11.5)
            )

        return self.async_show_form(
            step_id="add_user_body_metrics",
            data_schema=vol.Schema(schema),
        )

    async def async_step_edit_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Select user to edit."""
        if user_input is not None:
            selected = user_input["user_id"]
            if selected == "__legacy__":
                selected = ""  # map back to real legacy ID
            # Store selected user ID and proceed to edit details
            self.context["selected_user_id"] = selected
            return await self.async_step_edit_user_details()

        # Build user selection
        user_options: dict[str, str] = {}
        legacy_placeholder = "__legacy__"
        for user in self.user_profiles:
            uid = user[CONF_USER_ID]
            if uid == "":
                user_options[legacy_placeholder] = user[CONF_USER_NAME]
            else:
                user_options[uid] = user[CONF_USER_NAME]

        return self.async_show_form(
            step_id="edit_user",
            data_schema=vol.Schema(
                {
                    vol.Required("user_id"): vol.In(user_options),
                }
            ),
        )

    async def async_step_edit_user_details(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Edit user details."""
        selected_user_id = self.context["selected_user_id"]

        # Find the user
        user_index = next(
            i
            for i, u in enumerate(self.user_profiles)
            if u[CONF_USER_ID] == selected_user_id
        )
        current_user = self.user_profiles[user_index]

        if user_input is not None:
            # Validate user_name is not empty and person entity is unique
            user_name = user_input[CONF_USER_NAME]
            person_entity = user_input.get(CONF_PERSON_ENTITY)

            errors = {}

            if not _validate_user_name_not_empty(user_name):
                errors["base"] = "empty_user_name"

            # Validate person entity is unique (excluding current user)
            if person_entity and not _validate_person_entity_unique(
                person_entity, self.user_profiles, exclude_user_id=selected_user_id
            ):
                errors["person_entity"] = "duplicate_person_entity"

            if errors:
                # Re-show form with errors (ordered: name → person link → mobile devices → features)
                current_person = current_user.get(CONF_PERSON_ENTITY)
                current_mobile_services = current_user.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                )
                available_mobile_services = _get_mobile_notify_services(self.hass)

                schema = {
                    vol.Required(
                        CONF_USER_NAME, default=current_user[CONF_USER_NAME]
                    ): str,
                }

                # Add person entity selector (with default only if one is set)
                if current_person:
                    schema[vol.Optional(CONF_PERSON_ENTITY, default=current_person)] = (
                        selector.EntitySelector(
                            selector.EntitySelectorConfig(domain="person")
                        )
                    )
                else:
                    schema[vol.Optional(CONF_PERSON_ENTITY)] = selector.EntitySelector(
                        selector.EntitySelectorConfig(domain="person")
                    )

                # Add mobile notify services selector if any are available
                if available_mobile_services:
                    schema[
                        vol.Optional(
                            CONF_MOBILE_NOTIFY_SERVICES, default=current_mobile_services
                        )
                    ] = cv.multi_select(available_mobile_services)

                # Add body metrics toggle last
                schema[
                    vol.Required(
                        CONF_BODY_METRICS_ENABLED,
                        default=current_user.get(CONF_BODY_METRICS_ENABLED, False),
                    )
                ] = cv.boolean

                return self.async_show_form(
                    step_id="edit_user_details",
                    data_schema=vol.Schema(schema),
                    errors=errors,
                )

            # Check if body metrics is being enabled/disabled
            body_metrics_enabled = user_input.get(CONF_BODY_METRICS_ENABLED, False)
            currently_enabled = current_user.get(CONF_BODY_METRICS_ENABLED, False)

            # If disabling body metrics, remove associated entities
            if not body_metrics_enabled and currently_enabled:
                _LOGGER.debug(
                    "Body metrics disabled for user %s, removing entities.",
                    selected_user_id,
                )
                entity_reg = er.async_get(self.hass)
                device_name = self.config_entry.title
                for desc in SENSOR_DESCRIPTIONS:
                    unique_id = get_sensor_unique_id(
                        device_name, selected_user_id, desc.key
                    )
                    if entity_id := entity_reg.async_get_entity_id(
                        "sensor", DOMAIN, unique_id
                    ):
                        _LOGGER.debug("Removing entity %s", entity_id)
                        entity_reg.async_remove(entity_id)

            if body_metrics_enabled:
                # Enabling or keeping body metrics enabled - proceed to body metrics step
                # This allows editing existing body metrics as well
                self.context["edit_user_input"] = user_input
                return await self.async_step_edit_user_body_metrics()

            # Update user profile (user_id is preserved via **current_user)
            updated_user = {
                **current_user,
                CONF_USER_NAME: user_name,
                CONF_PERSON_ENTITY: user_input.get(CONF_PERSON_ENTITY),
                CONF_MOBILE_NOTIFY_SERVICES: user_input.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                ),
                CONF_BODY_METRICS_ENABLED: body_metrics_enabled,
                CONF_UPDATED_AT: datetime.now().isoformat(),
            }

            # If disabling body metrics, remove those fields
            if not body_metrics_enabled:
                updated_user.pop(CONF_SEX, None)
                updated_user.pop(CONF_BIRTHDATE, None)
                updated_user.pop(CONF_HEIGHT, None)

            # Update profiles list
            updated_profiles = list(self.user_profiles)
            updated_profiles[user_index] = updated_user

            # Update config entry
            new_data = {**self.config_entry.data, CONF_USER_PROFILES: updated_profiles}
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )

            # Reload the integration
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)

            return self.async_create_entry(title="", data={})

        # Build schema (ordered: name → person link → mobile devices → features)
        current_person = current_user.get(CONF_PERSON_ENTITY)
        current_mobile_services = current_user.get(CONF_MOBILE_NOTIFY_SERVICES, [])
        available_mobile_services = _get_mobile_notify_services(self.hass)

        schema = {
            vol.Required(CONF_USER_NAME, default=current_user[CONF_USER_NAME]): str,
        }

        # Add person entity selector (with default only if one is set)
        if current_person:
            schema[vol.Optional(CONF_PERSON_ENTITY, default=current_person)] = (
                selector.EntitySelector(selector.EntitySelectorConfig(domain="person"))
            )
        else:
            schema[vol.Optional(CONF_PERSON_ENTITY)] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain="person")
            )

        # Add mobile notify services selector if any are available
        if available_mobile_services:
            schema[
                vol.Optional(
                    CONF_MOBILE_NOTIFY_SERVICES, default=current_mobile_services
                )
            ] = cv.multi_select(available_mobile_services)

        # Add body metrics toggle last
        schema[
            vol.Required(
                CONF_BODY_METRICS_ENABLED,
                default=current_user.get(CONF_BODY_METRICS_ENABLED, False),
            )
        ] = cv.boolean

        return self.async_show_form(
            step_id="edit_user_details",
            data_schema=vol.Schema(schema),
        )

    async def async_step_edit_user_body_metrics(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Edit body metrics for user."""
        selected_user_id = self.context["selected_user_id"]

        # Find the user
        user_index = next(
            i
            for i, u in enumerate(self.user_profiles)
            if u[CONF_USER_ID] == selected_user_id
        )
        current_user = self.user_profiles[user_index]

        if user_input is not None:
            # Convert height to cm based on display unit
            if self.display_unit == UnitOfMass.KILOGRAMS:
                height_cm = user_input[CONF_HEIGHT]
            else:
                # Imperial - convert to cm
                height_cm = round(
                    DistanceConverter.convert(
                        user_input[CONF_FEET],
                        UnitOfLength.FEET,
                        UnitOfLength.CENTIMETERS,
                    )
                    + DistanceConverter.convert(
                        user_input[CONF_INCHES],
                        UnitOfLength.INCHES,
                        UnitOfLength.CENTIMETERS,
                    )
                )

            # Get basic user info from context
            basic_info = self.context["edit_user_input"]

            # Update user profile with body metrics
            updated_user = {
                **current_user,
                CONF_USER_NAME: basic_info[CONF_USER_NAME],
                CONF_PERSON_ENTITY: basic_info.get(CONF_PERSON_ENTITY),
                CONF_MOBILE_NOTIFY_SERVICES: basic_info.get(
                    CONF_MOBILE_NOTIFY_SERVICES, []
                ),
                CONF_BODY_METRICS_ENABLED: True,
                CONF_SEX: user_input[CONF_SEX],
                CONF_BIRTHDATE: user_input[CONF_BIRTHDATE],
                CONF_HEIGHT: height_cm,
                CONF_UPDATED_AT: datetime.now().isoformat(),
            }

            # Update profiles list
            updated_profiles = list(self.user_profiles)
            updated_profiles[user_index] = updated_user

            # Update config entry
            new_data = {**self.config_entry.data, CONF_USER_PROFILES: updated_profiles}
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )

            # Reload the integration
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)

            return self.async_create_entry(title="", data={})

        # Build schema based on display unit
        schema = {
            vol.Required(CONF_SEX, default=current_user.get(CONF_SEX, "Male")): vol.In(
                ["Male", "Female"]
            ),
            vol.Required(
                CONF_BIRTHDATE, default=current_user.get(CONF_BIRTHDATE)
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.DATE,
                ),
            ),
        }

        if self.display_unit == UnitOfMass.KILOGRAMS:
            schema[
                vol.Required(CONF_HEIGHT, default=current_user.get(CONF_HEIGHT, 170))
            ] = vol.All(vol.Coerce(int), vol.Range(min=100, max=250))
        else:
            # Convert cm to feet/inches for display
            height_cm = current_user.get(CONF_HEIGHT, 170)
            total_inches = height_cm / 2.54
            feet = int(total_inches // 12)
            inches = round((total_inches % 12) * 2) / 2

            schema[vol.Required(CONF_FEET, default=feet)] = vol.All(
                vol.Coerce(int), vol.Range(min=3, max=8)
            )
            schema[vol.Required(CONF_INCHES, default=inches)] = vol.All(
                vol.Coerce(float), vol.Range(min=0, max=11.5)
            )

        return self.async_show_form(
            step_id="edit_user_body_metrics",
            data_schema=vol.Schema(schema),
        )

    async def async_step_remove_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Remove a user."""
        # Check if this is the last user - cannot remove
        if len(self.user_profiles) <= 1:
            return self.async_abort(reason="cannot_remove_last_user")

        if user_input is not None:
            selected_user_id = user_input["user_id"]
            if selected_user_id == "__legacy__":
                selected_user_id = ""  # map back to real legacy ID

            # Clean up entities for the user being removed
            entity_reg = er.async_get(self.hass)
            device_name = self.config_entry.title

            # Get all possible sensor keys
            all_sensor_keys = [desc.key for desc in SENSOR_DESCRIPTIONS]
            all_sensor_keys.extend(["weight", "impedance"])  # Add base sensors

            for sensor_key in all_sensor_keys:
                unique_id = get_sensor_unique_id(
                    device_name, selected_user_id, sensor_key
                )
                if entity_id := entity_reg.async_get_entity_id(
                    "sensor", DOMAIN, unique_id
                ):
                    _LOGGER.debug(
                        "Removing entity %s for user %s", entity_id, selected_user_id
                    )
                    entity_reg.async_remove(entity_id)

            # Remove the selected user from profiles
            updated_profiles = [
                user
                for user in self.user_profiles
                if user[CONF_USER_ID] != selected_user_id
            ]

            # Update config entry
            new_data = {**self.config_entry.data, CONF_USER_PROFILES: updated_profiles}
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )

            # Reload the integration
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)

            return self.async_create_entry(title="", data={})

        # Build user selection
        user_options: dict[str, str] = {}
        legacy_placeholder = "__legacy__"
        for user in self.user_profiles:
            uid = user[CONF_USER_ID]
            if uid == "":
                user_options[legacy_placeholder] = user[CONF_USER_NAME]
            else:
                user_options[uid] = user[CONF_USER_NAME]

        return self.async_show_form(
            step_id="remove_user",
            data_schema=vol.Schema(
                {
                    vol.Required("user_id"): vol.In(user_options),
                }
            ),
        )

    async def async_step_scale_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Change scale settings."""
        if user_input is not None:
            # Update settings
            new_data = {
                **self.config_entry.data,
                CONF_SCALE_DISPLAY_UNIT: user_input[CONF_SCALE_DISPLAY_UNIT],
            }
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )

            # Reload the integration
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)

            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="scale_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_SCALE_DISPLAY_UNIT, default=self.display_unit
                    ): vol.In(
                        {
                            UnitOfMass.KILOGRAMS: "Metric (kg)",
                            UnitOfMass.POUNDS: "Imperial (lbs)",
                        }
                    ),
                }
            ),
        )

    async def async_step_advanced_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Change advanced settings."""
        if user_input is not None:
            # Update settings
            new_data = {
                **self.config_entry.data,
                CONF_HISTORY_RETENTION_DAYS: user_input[CONF_HISTORY_RETENTION_DAYS],
                CONF_MAX_HISTORY_SIZE: user_input[CONF_MAX_HISTORY_SIZE],
                CONF_ENABLE_LIBRARY_LOGGING: user_input[CONF_ENABLE_LIBRARY_LOGGING],
            }
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )

            # Reload the integration
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)

            return self.async_create_entry(title="", data={})

        # Get current values
        current_library_logging = self.config_entry.data.get(
            CONF_ENABLE_LIBRARY_LOGGING, False
        )

        return self.async_show_form(
            step_id="advanced_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_HISTORY_RETENTION_DAYS,
                        default=self.history_retention_days,
                    ): vol.All(
                        vol.Coerce(int),
                        vol.Range(min=1, max=365),
                    ),
                    vol.Required(
                        CONF_MAX_HISTORY_SIZE,
                        default=self.max_history_size,
                    ): vol.All(
                        vol.Coerce(int),
                        vol.Range(min=10, max=1000),
                    ),
                    vol.Required(
                        CONF_ENABLE_LIBRARY_LOGGING,
                        default=current_library_logging,
                    ): bool,
                }
            ),
        )
