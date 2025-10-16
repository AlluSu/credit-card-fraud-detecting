import os
import json
import numpy as np

def convert_to_serializable(obj):
    """Recursively convert NumPy arrays in dicts/lists to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

def convert_to_numpy(obj):
    """Recursively convert lists back to NumPy arrays where appropriate."""
    if isinstance(obj, list):
        # Check if it’s a list of numbers or nested lists -> convert to np.array
        if all(isinstance(x, (int, float, list)) for x in obj):
            return np.array([convert_to_numpy(x) for x in obj])
        else:
            return [convert_to_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_numpy(v) for k, v in obj.items()}
    else:
        return obj




CACHE_FILE = "cached_data.json"

def cache_finder(name):
    """Retrieve a value from the cache, converting lists back to NumPy arrays."""
    # Ensure the JSON file exists
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "w") as f:
            json.dump({}, f)
    
    # Load the cache
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
    
    # Retrieve the item if it exists
    if name in cache:
        value = convert_to_numpy(cache[name])
        return True, value
    else:
        return False, None
    


def cacher(name, value):
    """Save a value under a given name in the cache, converting arrays to lists."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Convert any numpy arrays to lists before saving
    serializable_value = convert_to_serializable(value)
    
    # Save/update the value
    cache[name] = serializable_value

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

