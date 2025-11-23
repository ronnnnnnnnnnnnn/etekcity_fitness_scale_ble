"""Adaptive tolerance calculation for person detection.

This module provides statistical analysis of weight history to calculate
user-specific tolerance values for person detection. Uses time-windowed
exponential decay averaging and standard deviation with recency scaling.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from collections import defaultdict
from dataclasses import dataclass

from .const import (
    ADAPTIVE_TOLERANCE_MAX_MULTIPLIER,
    ADAPTIVE_TOLERANCE_MIN_MULTIPLIER,
    CONF_WEIGHT_HISTORY,
    DEFAULT_TOLERANCE_PERCENTAGE,
    HISTORY_RETENTION_DAYS,
    MAX_TOLERANCE_KG,
    MIN_MEASUREMENTS_FOR_ADAPTIVE,
    MIN_TOLERANCE_KG,
    RECENCY_SCALING_BASE,
    RECENCY_SCALING_MAX,
    RECENCY_SCALING_RATE,
    REFERENCE_WINDOW_DAYS,
    TOLERANCE_MULTIPLIER,
    VARIANCE_WINDOW_DAYS,
)

_LOGGER = logging.getLogger(__name__)

MAX_MEASUREMENTS_PER_DAY_FOR_TOLERANCE = 2


@dataclass(frozen=True)
class _MeasurementRecord:
    """Measurement cache entry for aggregation."""

    timestamp: datetime
    data: dict


def _limit_measurements_per_day(measurements: list[dict]) -> list[dict]:
    """Limit measurement density so rapid repeats don't skew calculations."""

    if len(measurements) <= MAX_MEASUREMENTS_PER_DAY_FOR_TOLERANCE:
        return measurements

    day_buckets: defaultdict[datetime.date, list[_MeasurementRecord]] = defaultdict(list)
    for measurement in measurements:
        timestamp = datetime.fromisoformat(measurement["timestamp"])
        day_buckets[timestamp.date()].append(
            _MeasurementRecord(timestamp=timestamp, data=measurement)
        )

    limited: list[_MeasurementRecord] = []
    trimmed_count = 0

    for items in day_buckets.values():
        if len(items) <= MAX_MEASUREMENTS_PER_DAY_FOR_TOLERANCE:
            limited.extend(items)
            continue

        # Keep measurement with minimum and maximum weight for the day
        min_record = min(items, key=lambda rec: rec.data["weight_kg"])
        max_record = max(items, key=lambda rec: rec.data["weight_kg"])

        limited.append(min_record)
        kept = 1
        if max_record.data is not min_record.data:
            limited.append(max_record)
            kept += 1

        trimmed_count += len(items) - kept

    if trimmed_count > 0:
        _LOGGER.debug(
            "Limited measurement density: removed %d intra-day measurement(s) for adaptive tolerance calculations",
            trimmed_count,
        )

    limited.sort(key=lambda rec: rec.timestamp)
    return [record.data for record in limited]


def calculate_base_tolerance(weight_kg: float) -> float:
    """Calculate base tolerance using hybrid percentage with bounds.

    Uses percentage-based calculation with absolute minimum and maximum
    bounds to ensure tolerance scales appropriately across weight ranges.

    Args:
        weight_kg: Reference weight in kg

    Returns:
        Base tolerance in kg

    Examples:
        >>> calculate_base_tolerance(50.0)
        2.0  # 4% = 2.0kg
        >>> calculate_base_tolerance(75.0)
        3.0  # 4% = 3.0kg
        >>> calculate_base_tolerance(100.0)
        4.0  # 4% = 4.0kg
        >>> calculate_base_tolerance(30.0)
        1.5  # 4% = 1.2kg, clamped to min
        >>> calculate_base_tolerance(150.0)
        5.0  # 4% = 6.0kg, clamped to max
    """
    percentage_tolerance = weight_kg * DEFAULT_TOLERANCE_PERCENTAGE
    return max(MIN_TOLERANCE_KG, min(percentage_tolerance, MAX_TOLERANCE_KG))


def calculate_reference_weight(
    measurements: list[dict], current_time: datetime
) -> float | None:
    """Calculate reference weight using exponential decay weighting.

    Uses measurements within REFERENCE_WINDOW_DAYS, with more recent
    measurements weighted higher via exponential decay. This provides
    a stable reference that tracks recent trends while smoothing noise.

    Args:
        measurements: List of measurement dicts with 'timestamp' and 'weight_kg' keys
        current_time: Current time for age calculation

    Returns:
        Weighted average weight in kg, or None if no valid measurements

    Examples:
        With measurements at 0, 3, 7 days ago (7-day window):
        - 0 days: weight = 1.0
        - 3.5 days: weight ≈ 0.5 (half-life)
        - 7 days: weight ≈ 0.25
        Result is weighted average: (75.5*1.0 + 75.2*0.5 + 75.0*0.25) / (1.0+0.5+0.25)
    """
    if not measurements:
        return None

    cutoff_time = current_time - timedelta(days=REFERENCE_WINDOW_DAYS)

    # Filter to reference window
    recent = [
        m for m in measurements if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
    ]

    # Limit measurement density to avoid skew from rapid repeats
    recent = _limit_measurements_per_day(recent)

    if not recent:
        # Fallback to most recent measurement outside window
        fallback_weight = measurements[-1]["weight_kg"]
        _LOGGER.debug(
            "No measurements within %d-day reference window, using most recent: %.2f kg",
            REFERENCE_WINDOW_DAYS,
            fallback_weight,
        )
        return fallback_weight

    # Calculate exponential decay weights
    total_weight = 0.0
    weighted_sum = 0.0
    half_life = REFERENCE_WINDOW_DAYS / 2  # Half-life at midpoint of window

    for measurement in recent:
        timestamp = datetime.fromisoformat(measurement["timestamp"])
        age_days = (current_time - timestamp).total_seconds() / 86400
        # Exponential decay: weight = exp(-age / half_life)
        decay_weight = math.exp(-age_days / half_life)

        weighted_sum += measurement["weight_kg"] * decay_weight
        total_weight += decay_weight

    if total_weight > 0:
        ref_weight = weighted_sum / total_weight
        _LOGGER.debug(
            "Calculated reference weight %.2f kg from %d measurements (exponential decay, %d-day window)",
            ref_weight,
            len(recent),
            REFERENCE_WINDOW_DAYS,
        )
        return ref_weight

    # Fallback to most recent in window
    fallback_weight = recent[-1]["weight_kg"]
    _LOGGER.debug(
        "Reference weight calculation fallback: %.2f kg (most recent in window)",
        fallback_weight,
    )
    return fallback_weight


def calculate_adaptive_tolerance(
    measurements: list[dict], current_time: datetime, base_tolerance_kg: float
) -> float:
    """Calculate adaptive tolerance using standard deviation.

    Uses measurements within VARIANCE_WINDOW_DAYS to calculate standard
    deviation, then applies confidence interval multiplier. Falls back
    to base tolerance if insufficient data. Bounds result relative to
    base tolerance (not absolute values).

    Args:
        measurements: List of measurement dicts with 'timestamp' and 'weight_kg' keys
        current_time: Current time for window filtering
        base_tolerance_kg: Base tolerance to use as fallback and for bounds

    Returns:
        Adaptive tolerance in kg, bounded to [0.5x, 1.5x] of base tolerance

    Examples:
        With 10 measurements, std dev = 0.8kg, base = 3.0kg:
        - Raw adaptive = 0.8 * 2.5 = 2.0kg
        - Min bound = 3.0 * 0.5 = 1.5kg
        - Max bound = 3.0 * 1.5 = 4.5kg
        - Result = 2.0kg (within bounds)

        With < 5 measurements:
        - Result = base_tolerance_kg (fallback)
    """
    cutoff_time = current_time - timedelta(days=VARIANCE_WINDOW_DAYS)

    # Filter to variance window
    recent_measurements = [
        m
        for m in measurements
        if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
    ]

    # Limit measurement density within the window
    recent_measurements = _limit_measurements_per_day(recent_measurements)

    recent = [m["weight_kg"] for m in recent_measurements]

    # Need minimum measurements for statistical validity
    if len(recent) < MIN_MEASUREMENTS_FOR_ADAPTIVE:
        _LOGGER.debug(
            "Insufficient measurements for adaptive tolerance (%d < %d required), using base tolerance: %.2f kg",
            len(recent),
            MIN_MEASUREMENTS_FOR_ADAPTIVE,
            base_tolerance_kg,
        )
        return base_tolerance_kg

    # Calculate standard deviation
    mean = sum(recent) / len(recent)
    variance = sum((w - mean) ** 2 for w in recent) / len(recent)
    std_dev = math.sqrt(variance)

    # Apply confidence interval multiplier
    adaptive = std_dev * TOLERANCE_MULTIPLIER

    # Bound relative to base tolerance (scales with weight)
    min_adaptive = base_tolerance_kg * ADAPTIVE_TOLERANCE_MIN_MULTIPLIER
    max_adaptive = base_tolerance_kg * ADAPTIVE_TOLERANCE_MAX_MULTIPLIER

    bounded_adaptive = max(min_adaptive, min(adaptive, max_adaptive))

    # Log calculation details
    if bounded_adaptive == min_adaptive:
        _LOGGER.debug(
            "Adaptive tolerance %.2f kg clamped to minimum (%.2f kg, %.0f%% of base %.2f kg) "
            "[std dev: %.2f kg, %d measurements in %d-day window]",
            bounded_adaptive,
            min_adaptive,
            ADAPTIVE_TOLERANCE_MIN_MULTIPLIER * 100,
            base_tolerance_kg,
            std_dev,
            len(recent),
            VARIANCE_WINDOW_DAYS,
        )
    elif bounded_adaptive == max_adaptive:
        _LOGGER.debug(
            "Adaptive tolerance %.2f kg clamped to maximum (%.2f kg, %.0f%% of base %.2f kg) "
            "[std dev: %.2f kg, %d measurements in %d-day window]",
            bounded_adaptive,
            max_adaptive,
            ADAPTIVE_TOLERANCE_MAX_MULTIPLIER * 100,
            base_tolerance_kg,
            std_dev,
            len(recent),
            VARIANCE_WINDOW_DAYS,
        )
    else:
        _LOGGER.debug(
            "Adaptive tolerance %.2f kg from std dev %.2f kg × %.1fx multiplier "
            "[%d measurements in %d-day window, base: %.2f kg]",
            bounded_adaptive,
            std_dev,
            TOLERANCE_MULTIPLIER,
            len(recent),
            VARIANCE_WINDOW_DAYS,
            base_tolerance_kg,
        )

    return bounded_adaptive


def calculate_recency_multiplier(days_since_last: float) -> float:
    """Calculate tolerance multiplier based on staleness.

    Uses square root scaling for sub-linear growth, capped at maximum.
    This increases tolerance for stale data without growing unboundedly.

    Args:
        days_since_last: Days since user's last measurement

    Returns:
        Multiplier >= 1.0, capped at RECENCY_SCALING_MAX

    Examples:
        >>> calculate_recency_multiplier(0)
        1.0  # No scaling
        >>> calculate_recency_multiplier(7)
        1.40  # ~40% increase
        >>> calculate_recency_multiplier(14)
        1.56  # ~56% increase
        >>> calculate_recency_multiplier(30)
        1.82  # ~82% increase
        >>> calculate_recency_multiplier(60)
        2.16  # ~116% increase
        >>> calculate_recency_multiplier(90)
        2.42  # ~142% increase
        >>> calculate_recency_multiplier(100)
        2.5  # Capped at max
        >>> calculate_recency_multiplier(365)
        2.5  # Still capped
    """
    if days_since_last <= 0:
        return RECENCY_SCALING_BASE

    # Square root scaling: grows quickly initially, then slows
    multiplier = RECENCY_SCALING_BASE + (
        RECENCY_SCALING_RATE * math.sqrt(days_since_last)
    )
    return min(multiplier, RECENCY_SCALING_MAX)


def get_tolerance_for_user(
    user_profile: dict, current_time: datetime
) -> tuple[float | None, float | None]:
    """Calculate final tolerance for a user.

    Combines base tolerance, adaptive tolerance, and recency scaling.
    Returns (None, None) for users with no usable history (new users
    or stale data beyond retention window).

    Args:
        user_profile: User profile dict with CONF_WEIGHT_HISTORY key
        current_time: Current timestamp

    Returns:
        Tuple of (reference_weight_kg, final_tolerance_kg) or (None, None)

    Examples:
        New user (no history):
            >>> get_tolerance_for_user({CONF_WEIGHT_HISTORY: []}, datetime.now())
            (None, None)

        User with recent measurements:
            - Reference weight: 75.0kg
            - Base tolerance: 3.0kg (4% of 75kg)
            - Adaptive tolerance: 2.5kg (user's std dev * 2.5)
            - Days since last: 7
            - Recency multiplier: 1.4x
            - Final: max(1.5, 2.5 * 1.4) = 3.5kg
            >>> # Returns: (75.0, 3.5)

        User with stale history (90+ days):
            >>> get_tolerance_for_user({...}, current_time)
            (None, None)  # Treated as new user
    """
    history = user_profile.get(CONF_WEIGHT_HISTORY, [])
    user_id = user_profile.get("user_id", "unknown")
    user_name = user_profile.get("name", user_id)

    if not history:
        # No history - treat as new user
        _LOGGER.debug(
            "User %s has no weight history - cannot calculate tolerance (new user)",
            user_name,
        )
        return (None, None)

    # Check if history is too stale
    last_measurement_time = datetime.fromisoformat(history[-1]["timestamp"])
    days_since_last = (current_time - last_measurement_time).total_seconds() / 86400

    if days_since_last > HISTORY_RETENTION_DAYS:
        # Stale history - treat as new user
        _LOGGER.debug(
            "User %s has stale weight history (%.1f days > %d day retention) - cannot calculate tolerance",
            user_name,
            days_since_last,
            HISTORY_RETENTION_DAYS,
        )
        return (None, None)

    _LOGGER.debug(
        "Calculating adaptive tolerance for user %s (%.1f days since last measurement, %d total measurements)",
        user_name,
        days_since_last,
        len(history),
    )

    # Calculate reference weight (weighted average)
    ref_weight = calculate_reference_weight(history, current_time)

    if ref_weight is None:
        _LOGGER.warning("Failed to calculate reference weight for user %s", user_name)
        return (None, None)

    # Calculate base tolerance (hybrid percentage)
    base_tolerance = calculate_base_tolerance(ref_weight)
    _LOGGER.debug(
        "Base tolerance for user %s: %.2f kg (%.0f%% of reference weight %.2f kg)",
        user_name,
        base_tolerance,
        DEFAULT_TOLERANCE_PERCENTAGE * 100,
        ref_weight,
    )

    # Calculate adaptive tolerance (statistical)
    adaptive_tolerance = calculate_adaptive_tolerance(
        history, current_time, base_tolerance
    )

    # Apply recency scaling
    recency_mult = calculate_recency_multiplier(days_since_last)
    if recency_mult > RECENCY_SCALING_BASE:
        _LOGGER.debug(
            "Applying recency multiplier %.2fx for user %s (%.1f days since last measurement)",
            recency_mult,
            user_name,
            days_since_last,
        )

    # Final tolerance (only apply lower bound, no upper clamp)
    final_tolerance = max(MIN_TOLERANCE_KG, adaptive_tolerance * recency_mult)

    _LOGGER.debug(
        "Final tolerance for user %s: %.2f kg [reference: %.2f kg, adaptive: %.2f kg, "
        "recency: %.2fx, final: %.2f kg]",
        user_name,
        final_tolerance,
        ref_weight,
        adaptive_tolerance,
        recency_mult,
        final_tolerance,
    )

    return (ref_weight, final_tolerance)
