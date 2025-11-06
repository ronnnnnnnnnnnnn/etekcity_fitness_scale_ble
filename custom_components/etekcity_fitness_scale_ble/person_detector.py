"""Person detection based on weight measurements."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.helpers import entity_registry as er

from .const import CONF_PERSON_ENTITY, get_sensor_unique_id

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Weight tolerance in kg for person detection (±3kg)
WEIGHT_TOLERANCE_KG = 3.0


class PersonDetector:
    """Detect which person is using the scale based on weight."""

    def __init__(self, hass: HomeAssistant, device_name: str, domain: str) -> None:
        """Initialize the person detector.

        Args:
            hass: Home Assistant instance for accessing sensor states.
            device_name: The device name used for unique_id construction.
            domain: The integration domain.
        """
        self.hass = hass
        self._device_name = device_name
        self._domain = domain
        # Cache entity registry for performance (avoid lookup on every detection)
        self._entity_reg = er.async_get(hass)

    def _filter_candidates_by_location(
        self, candidate_ids: list[str], user_profiles: list[dict]
    ) -> list[str]:
        """Filter a list of candidates by their Home Assistant person entity state.

        Args:
            candidate_ids: A list of user_ids to filter.
            user_profiles: The full list of user profiles to look up person entities.

        Returns:
            A filtered list of user_ids, excluding anyone who is 'not_home'.
            If filtering results in an empty list, returns the original list.
        """
        # Create a quick lookup map for profiles
        profiles_by_id = {p.get("user_id"): p for p in user_profiles}

        filtered_candidates = []
        for user_id in candidate_ids:
            profile = profiles_by_id.get(user_id)
            if not profile:
                continue

            person_entity_id = profile.get(CONF_PERSON_ENTITY)
            if not person_entity_id:
                # If no person is linked, we can't filter, so keep them
                filtered_candidates.append(user_id)
                continue

            person_state = self.hass.states.get(person_entity_id)
            if not person_state:
                _LOGGER.debug(
                    "Person entity %s not found for user %s, keeping as candidate.",
                    person_entity_id,
                    user_id,
                )
                filtered_candidates.append(user_id)
                continue

            if person_state.state.lower() == "not_home":
                _LOGGER.debug(
                    "Excluding user %s from candidates because they are not home (state: %s)",
                    user_id,
                    person_state.state,
                )
                continue  # Exclude this user

            # Keep the user if they are home or state is not 'not_home'
            filtered_candidates.append(user_id)

        # Edge case: if filtering removed everyone, return the original list
        if not filtered_candidates and candidate_ids:
            _LOGGER.debug(
                "Location filter removed all candidates; falling back to original list to prevent data loss."
            )
            return candidate_ids

        return filtered_candidates

    def detect_person(
        self, weight_kg: float, user_profiles: list[dict]
    ) -> tuple[str | None, list[str]]:
        """Detect which person is using the scale based on weight.

        Uses a simple tolerance-based algorithm: checks if the current weight
        is within ±3kg of the last known weight for each user.

        Args:
            weight_kg: Current weight measurement in kilograms.
            user_profiles: List of user profile dictionaries with user_id.

        Returns:
            A tuple of (detected_user_id, ambiguous_user_ids).
            - If exactly one user matches: (user_id, [])
            - If multiple users match: (None, [user_id1, user_id2, ...])
            - If no users match: (None, [])
        """
        if not user_profiles:
            _LOGGER.debug("No user profiles configured, cannot detect person")
            return (None, [])

        matching_users = []

        for user_profile in user_profiles:
            user_id = user_profile.get("user_id")
            if not user_id:
                continue

            # Construct unique_id for weight sensor using helper function
            sensor_unique_id = get_sensor_unique_id(
                self._device_name, user_id, "weight"
            )

            # Look up entity_id from unique_id via entity registry
            sensor_entity_id = self._entity_reg.async_get_entity_id(
                "sensor", self._domain, sensor_unique_id
            )

            if not sensor_entity_id:
                _LOGGER.debug(
                    "No weight sensor found in registry for user %s (unique_id: %s)",
                    user_profile.get("name", user_id),
                    sensor_unique_id,
                )
                continue

            # Get sensor state
            sensor_state = self.hass.states.get(sensor_entity_id)

            if not sensor_state or sensor_state.state in ("unknown", "unavailable"):
                _LOGGER.debug(
                    "No previous weight found for user %s (sensor: %s)",
                    user_profile.get("name", user_id),
                    sensor_entity_id,
                )
                continue

            try:
                last_weight_kg = float(sensor_state.state)
            except (ValueError, TypeError):
                _LOGGER.warning(
                    "Invalid weight value for user %s: %s",
                    user_profile.get("name", user_id),
                    sensor_state.state,
                )
                continue

            # Check if current weight is within tolerance
            weight_diff = abs(weight_kg - last_weight_kg)
            if weight_diff <= WEIGHT_TOLERANCE_KG:
                _LOGGER.debug(
                    "User %s matches (last: %.2f kg, current: %.2f kg, diff: %.2f kg)",
                    user_profile.get("name", user_id),
                    last_weight_kg,
                    weight_kg,
                    weight_diff,
                )
                matching_users.append(user_id)
            else:
                _LOGGER.debug(
                    "User %s does not match (last: %.2f kg, current: %.2f kg, diff: %.2f kg > %.2f kg)",
                    user_profile.get("name", user_id),
                    last_weight_kg,
                    weight_kg,
                    weight_diff,
                    WEIGHT_TOLERANCE_KG,
                )

        # Return results based on number of matches
        if len(matching_users) == 1:
            _LOGGER.debug(
                "Detected person: %s (weight: %.2f kg)",
                matching_users[0],
                weight_kg,
            )
            return (matching_users[0], [])
        elif len(matching_users) > 1:
            _LOGGER.debug(
                "Ambiguous detection: %d users match (weight: %.2f kg): %s",
                len(matching_users),
                weight_kg,
                matching_users,
            )
            return (None, matching_users)
        else:
            _LOGGER.debug(
                "No matching user found for weight: %.2f kg",
                weight_kg,
            )
            return (None, [])

    def get_users_with_history(self, user_profiles: list[dict]) -> list[str]:
        """Get list of user IDs that have previous weight measurements.

        Args:
            user_profiles: List of user profile dictionaries with user_id.

        Returns:
            List of user IDs that have weight sensor history.
        """
        users_with_history = []

        for user_profile in user_profiles:
            user_id = user_profile.get("user_id")
            if not user_id:
                continue

            # Construct unique_id for weight sensor
            sensor_unique_id = get_sensor_unique_id(
                self._device_name, user_id, "weight"
            )

            # Look up entity_id
            sensor_entity_id = self._entity_reg.async_get_entity_id(
                "sensor", self._domain, sensor_unique_id
            )

            if not sensor_entity_id:
                continue

            # Check if sensor has a valid state
            sensor_state = self.hass.states.get(sensor_entity_id)
            if sensor_state and sensor_state.state not in ("unknown", "unavailable"):
                try:
                    # Try to parse as float to verify it's a valid measurement
                    float(sensor_state.state)
                    users_with_history.append(user_id)
                except (ValueError, TypeError):
                    # Invalid state value
                    continue

        return users_with_history
