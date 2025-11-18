"""Constants for the etekcity_fitness_scale_ble integration."""

DOMAIN = "etekcity_fitness_scale_ble"

# Legacy constants (used for migration from v1)
CONF_CALC_BODY_METRICS = "calculate body metrics"
CONF_SEX = "sex"
CONF_HEIGHT = "height"
CONF_BIRTHDATE = "birthdate"
CONF_FEET = "feet"
CONF_INCHES = "inches"

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

# Reserved user_id for v1 compatibility
# Empty string preserves v1 entity_ids during migration to v2 multi-user format
V1_LEGACY_USER_ID = ""


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
