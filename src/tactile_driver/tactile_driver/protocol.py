import json
import math


def parse_frame(line, count):
    """Validate a firmware JSON frame before publishing float32 taxels."""
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError('frame must be an object')
    values = payload.get('values')
    if not isinstance(values, list) or len(values) != count:
        raise ValueError('incorrect taxel count')
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or abs(v) > 3.402823466e38 for v in values):
        raise ValueError('taxels must be finite float32 numbers')
    return [float(v) for v in values]
