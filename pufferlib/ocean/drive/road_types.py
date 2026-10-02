"""Road rules and binary type flags shared by map and observation writers."""

NONE = 0
STOP_SIGN = 7
TRAFFIC_LIGHT_STOP_LINE = 9
RIGHT_HAND_STOP_LINE = 11
TRAFFIC_LIGHT_YIELD_FLAG = 0x200  # Must match drive.h; adds no record payload.
ENTITY_TYPE_MASK = 0xFF
YIELD_SUBTYPES = frozenset({"yield", "yield_manual", "yield_stop"})


def stop_line_fallback_type(attributes):
    """An unlit stop line follows its underlying unsignalized road rule."""
    return STOP_SIGN if "subtype" in attributes and attributes["subtype"] in YIELD_SUBTYPES else NONE
