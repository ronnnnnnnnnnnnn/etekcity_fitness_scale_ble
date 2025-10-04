"""Constants for the etekcity_fitness_scale_ble integration."""

from enum import StrEnum

DOMAIN = "etekcity_fitness_scale_ble"

CONF_CALC_BODY_METRICS = "calculate body metrics"
CONF_SEX = "sex"
CONF_HEIGHT = "height"
CONF_BIRTHDATE = "birthdate"

CONF_FEET = "feet"
CONF_INCHES = "inches"

CONF_SCALE_MODEL = "scale_model"


class ScaleModel(StrEnum):
    """Scale model enumeration."""

    ESF551 = "ESF-551"
    ESF24 = "ESF-24"


# Detection patterns for different scale models
SCALE_DETECTION_PATTERNS = {
    ScaleModel.ESF24: {
        "local_name": "QN-Scale1"
    },
    ScaleModel.ESF551: {
        "local_name_pattern": "Etekcity *Fitness *Scale*",
        "manufacturer_id": 1744
    }
}
