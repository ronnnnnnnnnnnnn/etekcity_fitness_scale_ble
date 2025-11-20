"""Adaptive tolerance calculation for person detection.

This module provides statistical analysis of weight history to calculate
user-specific tolerance values for person detection. Uses time-windowed
exponential decay averaging and standard deviation with recency scaling.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

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

    if not recent:
        # Fallback to most recent measurement outside window
        return measurements[-1]["weight_kg"]

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
        return weighted_sum / total_weight

    # Fallback to most recent in window
    return recent[-1]["weight_kg"]


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
    recent = [
        m["weight_kg"]
        for m in measurements
        if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
    ]

    # Need minimum measurements for statistical validity
    if len(recent) < MIN_MEASUREMENTS_FOR_ADAPTIVE:
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

    return max(min_adaptive, min(adaptive, max_adaptive))


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

    if not history:
        # No history - treat as new user
        return (None, None)

    # Check if history is too stale
    last_measurement_time = datetime.fromisoformat(history[-1]["timestamp"])
    days_since_last = (current_time - last_measurement_time).total_seconds() / 86400

    if days_since_last > HISTORY_RETENTION_DAYS:
        # Stale history - treat as new user
        return (None, None)

    # Calculate reference weight (weighted average)
    ref_weight = calculate_reference_weight(history, current_time)

    if ref_weight is None:
        return (None, None)

    # Calculate base tolerance (hybrid percentage)
    base_tolerance = calculate_base_tolerance(ref_weight)

    # Calculate adaptive tolerance (statistical)
    adaptive_tolerance = calculate_adaptive_tolerance(
        history, current_time, base_tolerance
    )

    # Apply recency scaling
    recency_mult = calculate_recency_multiplier(days_since_last)

    # Final tolerance (only apply lower bound, no upper clamp)
    final_tolerance = max(MIN_TOLERANCE_KG, adaptive_tolerance * recency_mult)

    return (ref_weight, final_tolerance)
