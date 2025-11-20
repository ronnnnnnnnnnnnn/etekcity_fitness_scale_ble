"""Person detection based on weight measurements."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING

from .adaptive_tolerance import get_tolerance_for_user
from .const import CONF_PERSON_ENTITY

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class PersonDetector:
    """Detect which person is using the scale based on weight."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the person detector.

        Args:
            hass: Home Assistant instance for accessing person entity states.
        """
        self.hass = hass

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

        Uses adaptive tolerance based on each user's weight history, variance,
        and time since last measurement. Falls back to treating users with no
        history or stale history (90+ days) as new users.

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

        current_time = datetime.now()
        matching_users = []

        for user_profile in user_profiles:
            user_id = user_profile.get("user_id")
            user_name = user_profile.get("name", user_id)
            if user_id is None:
                continue

            # Calculate adaptive tolerance for this user
            ref_weight, tolerance_kg = get_tolerance_for_user(
                user_profile, current_time
            )

            # If tolerance is None, user has no usable history (new user or 90+ day gap)
            if tolerance_kg is None:
                _LOGGER.debug(
                    "User %s has no usable weight history (new user or stale data), skipping tolerance check",
                    user_name,
                )
                continue

            # Check if current weight is within adaptive tolerance
            weight_diff = abs(weight_kg - ref_weight)
            if weight_diff <= tolerance_kg:
                _LOGGER.debug(
                    "User %s matches (ref: %.2f kg, current: %.2f kg, diff: %.2f kg, tolerance: %.2f kg)",
                    user_name,
                    ref_weight,
                    weight_kg,
                    weight_diff,
                    tolerance_kg,
                )
                matching_users.append(user_id)
            else:
                _LOGGER.debug(
                    "User %s does not match (ref: %.2f kg, current: %.2f kg, diff: %.2f kg > tolerance: %.2f kg)",
                    user_name,
                    ref_weight,
                    weight_kg,
                    weight_diff,
                    tolerance_kg,
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
        """Get list of user IDs that have usable weight history.

        A user has usable history if they have weight_history data that can be used
        to calculate adaptive tolerance (not None). This excludes new users and users
        with stale data (90+ days old).

        Args:
            user_profiles: List of user profile dictionaries with user_id.

        Returns:
            List of user IDs that have usable weight history for adaptive tolerance.
        """
        users_with_history = []
        current_time = datetime.now()

        for user_profile in user_profiles:
            user_id = user_profile.get("user_id")
            if user_id is None:
                continue

            # Check if user has usable weight history via adaptive tolerance
            ref_weight, tolerance_kg = get_tolerance_for_user(
                user_profile, current_time
            )

            # If tolerance is not None, user has usable history
            if tolerance_kg is not None:
                users_with_history.append(user_id)

        return users_with_history
