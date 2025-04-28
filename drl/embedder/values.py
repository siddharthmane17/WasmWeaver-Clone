from core.value import Val, I32, I64, F32, F64, RefFunc

MAX_VALUE_TYPE_INDEX = 5

def embedd_value_type(value: Val):
    if isinstance(value, I32):
        return 1
    elif isinstance(value, I64):
        return 2
    elif isinstance(value, F32):
        return 3
    elif isinstance(value, F64):
        return 4
    elif isinstance(value, RefFunc):
        return 5
    raise ValueError(f"Unknown value type: {type(value)}")
