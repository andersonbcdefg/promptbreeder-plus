import re
import json
import json5

def get_from_json(json_string, key, fallback=None):
    """
    Get a value from JSON, with increasingly permissive/noisy fallbacks.
    1. Parse as JSON, extract the key.
    2. Use JSON5 to parse, extract the key.
    3. Use regexp to find the key and extract whatever value follows it.
    """
    try:
        # Try parsing with standard JSON
        data = json.loads(json_string)
        if key in data:
            return data[key]
    except json.JSONDecodeError:
        pass

    try:
        # Try parsing with JSON5
        data = json5.loads(json_string)
        if key in data:
            return data[key]
    except Exception as e:
        pass

    # Try using regexp to find the key and extract whatever value follows it
    match = re.search(f'"{key}"\s*:\s*"(.*?)(?:"|$)', json_string)
    if match:
        return match.group(1)
    
    # otherwise get whatever is after the colon but before a comma
    match = re.search(f'"{key}"\s*:\s*(.*?),', json_string)
    if match:
        return match.group(1)
    
    return fallback
    
