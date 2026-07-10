"""Constants for the etekcity_fitness_scale_ble integration."""

from etekcity_esf551_ble import CAPABILITIES, ScaleModel as ScaleModel

DOMAIN = "etekcity_fitness_scale_ble"

# Legacy constants (used for migration from v1)
CONF_CALC_BODY_METRICS = "calculate body metrics"
CONF_SEX = "sex"
CONF_HEIGHT = "height"
CONF_BIRTHDATE = "birthdate"
CONF_FEET = "feet"
CONF_INCHES = "inches"

CONF_SCALE_MODEL = "scale_model"


# Model detection lives in the etekcity_esf551_ble library (detect_model,
# MODEL_CODES, FALLBACK_MATCHERS). manifest.json's "bluetooth" array only
# decides which advertisements wake the discovery flow; classification is
# done by the library, which confirms or rejects each device.
# ScaleModel is re-exported here because its string values are persisted in
# config entries; the library treats those values as a stable contract.

# Models that measure impedance and therefore support body-metrics
# calculation (weight + impedance + user profile).
BODY_METRICS_MODELS = frozenset(
    model for model, caps in CAPABILITIES.items() if caps.has_impedance
)

# Models that also report heart rate (bpm) on the final measurement.
HEART_RATE_MODELS = frozenset(
    model for model, caps in CAPABILITIES.items() if caps.has_heart_rate
)

# Multi-user constants (v2+)
CONF_SCALE_DISPLAY_UNIT = "scale_display_unit"
CONF_USER_PROFILES = "user_profiles"

# User profile keys
CONF_USER_ID = "user_id"
CONF_USER_NAME = "name"
CONF_PERSON_ENTITY = "person_entity"
CONF_MOBILE_NOTIFY_SERVICES = "mobile_notify_services"
CONF_BODY_METRICS_ENABLED = "body_metrics_enabled"
CONF_CREATED_AT = "created_at"
CONF_UPDATED_AT = "updated_at"
CONF_WEIGHT_HISTORY = "weight_history"

# Reserved user_id for v1 compatibility
# Empty string preserves v1 entity_ids during migration to v2 multi-user format
V1_LEGACY_USER_ID = ""

# Weight history tracking (v3+)
HISTORY_RETENTION_DAYS = 90  # Maximum age of measurements to retain (default)
MAX_HISTORY_SIZE = 100  # Maximum number of measurements per user (default)

# Configurable history settings (v4+)
CONF_HISTORY_RETENTION_DAYS = "history_retention_days"
CONF_MAX_HISTORY_SIZE = "max_history_size"

# Advanced settings
CONF_ENABLE_LIBRARY_LOGGING = "enable_library_logging"

# Adaptive tolerance - base calculation (hybrid percentage with bounds)
DEFAULT_TOLERANCE_PERCENTAGE = 0.04  # 4% of user's weight
MIN_TOLERANCE_KG = 1.5  # Minimum absolute tolerance
MAX_TOLERANCE_KG = 5.0  # Maximum absolute tolerance

# Adaptive tolerance - time windows
REFERENCE_WINDOW_DAYS = 7  # Weighted average window for reference weight
VARIANCE_WINDOW_DAYS = 30  # Sample window for standard deviation calculation
MIN_MEASUREMENTS_FOR_ADAPTIVE = 5  # Minimum data points for statistical validity

# Adaptive tolerance - calculation parameters
TOLERANCE_MULTIPLIER = (
    2.5  # Std dev multiplier (≈98.8% confidence for normal distribution)
)
ADAPTIVE_TOLERANCE_MIN_MULTIPLIER = 0.5  # Adaptive tolerance >= 50% of base
ADAPTIVE_TOLERANCE_MAX_MULTIPLIER = 1.5  # Adaptive tolerance <= 150% of base

# Adaptive tolerance - recency scaling (sub-linear growth)
RECENCY_SCALING_BASE = 1.0  # Base multiplier (no scaling for fresh data)
RECENCY_SCALING_RATE = 0.15  # Rate coefficient for sqrt scaling
RECENCY_SCALING_MAX = 2.5  # Maximum tolerance multiplier for stale data


def get_sensor_unique_id(device_name: str, user_id: str, sensor_key: str) -> str:
    """Construct unique_id for a sensor entity.

    Handles v1 legacy user_id (empty string) to preserve entity identity
    during migration from single-user to multi-user format.

    Args:
        device_name: The device name from entry.title (e.g., "Etekcity ESF551 E1A2B3").
        user_id: The user ID. Empty string "" for v1 compatibility, UUID for v2+.
        sensor_key: The sensor key (e.g., "weight", "impedance", "body_fat_percentage").

    Returns:
        The unique_id string used for entity registry.

    Examples:
        >>> get_sensor_unique_id("Scale ABC", "", "weight")
        "Scale ABC_weight"
        >>> get_sensor_unique_id("Scale ABC", "uuid123", "weight")
        "Scale ABC_uuid123_weight"
    """
    if user_id == V1_LEGACY_USER_ID:
        # v1 compatibility - preserve original entity identity
        return f"{device_name}_{sensor_key}"
    # v2 multi-user format
    return f"{device_name}_{user_id}_{sensor_key}"


def parse_notify_service(stored: str) -> tuple[str, str]:
    """Normalize a stored notify-service value into (domain, name).

    The config flow stores the short form (e.g. "mobile_app_pixel_9a") and
    the coordinator hardcodes the "notify" domain when sending. Tolerating
    both forms here keeps the lookup, any existence checks, and the actual
    services.async_call consistent if a stored value ever carries the
    "notify." prefix.
    """
    if "." in stored:
        domain, name = stored.split(".", 1)
        return domain, name
    return "notify", stored
