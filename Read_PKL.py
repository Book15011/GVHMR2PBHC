import pickle
import joblib
import numpy as np
import sys
from collections.abc import Iterable
from types import SimpleNamespace
from collections import defaultdict

# Define simple types that don't need recursion
SIMPLE_TYPES = (int, float, str, bool, bytes, type(None), complex)

def get_type_details(obj):
    """Get detailed type information about an object"""
    obj_type = type(obj)
    type_name = obj_type.__name__
    module = obj_type.__module__
    
    if module not in ('builtins', '__builtin__'):
        type_name = f"{module}.{type_name}"
    
    return type_name

def get_container_uniformity(items):
    """Check if container items have uniform type"""
    if not items:
        return None
    
    first_type = type(items[0])
    if all(type(item) is first_type for item in items):
        return get_type_details(items[0])
    return "mixed"

def explore_structure(obj, depth=0, max_depth=4, max_length=6, visited=None):
    """
    Recursively explores object structure with detailed metadata
    
    Returns:
    - A tuple (output_lines, type_counter) containing:
      output_lines: List of text lines describing the structure
      type_counter: Dict counting occurrences of each type
    """
    if visited is None:
        visited = set()
    
    output = []
    type_counter = defaultdict(int)
    
    def add_line(text):
        output.append(("  " * depth) + text)
    
    # Handle recursion limits and cycles
    obj_id = id(obj)
    if depth > max_depth:
        type_name = get_type_details(obj)
        add_line(f"{type_name}: ... (max_depth reached)")
        type_counter[type_name] += 1
        return output, type_counter
    
    if obj_id in visited:
        add_line(f"{get_type_details(obj)}: <already visited>")
        return output, type_counter
    
    visited.add(obj_id)
    
    # Get object type information
    obj_type = type(obj)
    type_name = get_type_details(obj)
    type_counter[type_name] += 1
    
    # Handle simple types
    if isinstance(obj, SIMPLE_TYPES):
        add_line(f"{type_name}: {repr(obj)}")
        return output, type_counter
    
    # Handle dictionaries
    if isinstance(obj, dict):
        type_info = f"dict ({len(obj)} items)"
        uniformity = get_container_uniformity(list(obj.values())[:max_length])
        if uniformity:
            type_info += f" [value type: {uniformity}]"
        add_line(type_info)
        
        for i, (key, value) in enumerate(obj.items()):
            if i >= max_length:
                add_line(f"... {len(obj) - max_length} more keys")
                break
                
            key_type = get_type_details(key)
            add_line(f"[Key {i}]: {key_type} = {repr(key)}")
            
            child_out, child_count = explore_structure(
                value, depth+1, max_depth, max_length, visited
            )
            output.extend(child_out)
            for k, v in child_count.items():
                type_counter[k] += v
                
        return output, type_counter
    
    # Handle sequences (list, tuple, set)
    if isinstance(obj, (list, tuple, set)):
        typename = type(obj).__name__
        type_info = f"{typename} ({len(obj)} items)"
        
        # Check uniformity of elements
        if obj:
            uniformity = get_container_uniformity(list(obj)[:max_length])
            if uniformity:
                type_info += f" [element type: {uniformity}]"
        
        add_line(type_info)
        
        for i, item in enumerate(obj):
            if i >= max_length:
                add_line(f"... {len(obj) - max_length} more items")
                break
                
            add_line(f"[{i}]:")
            child_out, child_count = explore_structure(
                item, depth+1, max_depth, max_length, visited
            )
            output.extend(child_out)
            for k, v in child_count.items():
                type_counter[k] += v
                
        return output, type_counter
    
    # Handle NumPy arrays
    if type(obj).__name__ == 'ndarray':
        details = (f"shape={obj.shape}, dtype={obj.dtype}, "
                  f"min={np.nanmin(obj):.4f}, max={np.nanmax(obj):.4f}")
        add_line(f"numpy.ndarray: {details}")
        return output, type_counter
    
    # Handle Pandas DataFrames and Series
    if 'pandas' in sys.modules:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            cols = ", ".join(obj.columns.astype(str))
            add_line(f"DataFrame ({obj.shape[0]} rows × {obj.shape[1]} cols)")
            add_line(f"Columns: [{cols}]")
            add_line(f"Dtypes:\n{obj.dtypes}")
            return output, type_counter
        elif isinstance(obj, pd.Series):
            add_line(f"Series ({len(obj)} items) [dtype: {obj.dtype}]")
            return output, type_counter
    
    # Handle custom objects
    if hasattr(obj, '__dict__') or isinstance(obj, SimpleNamespace):
        try:
            attrs = vars(obj)
            add_line(f"Instance of {type_name} with {len(attrs)} attributes:")
            
            for attr, value in list(attrs.items())[:max_length]:
                add_line(f".{attr}: {get_type_details(value)}")
                child_out, child_count = explore_structure(
                    value, depth+1, max_depth, max_length, visited
                )
                output.extend(child_out)
                for k, v in child_count.items():
                    type_counter[k] += v
                    
            if len(attrs) > max_length:
                add_line(f"... {len(attrs) - max_length} more attributes")
        except Exception as e:
            add_line(f"[Error reading attributes: {str(e)}]")
            
        return output, type_counter
    
    # Handle other iterables
    if isinstance(obj, Iterable):
        add_line(f"Iterable: {type_name}")
        try:
            for i, item in enumerate(obj):
                if i >= max_length:
                    add_line(f"... (truncated after {max_length} items)")
                    break
                    
                add_line(f"Item {i}:")
                child_out, child_count = explore_structure(
                    item, depth+1, max_depth, max_length, visited
                )
                output.extend(child_out)
                for k, v in child_count.items():
                    type_counter[k] += v
        except Exception as e:
            add_line(f"[Error iterating: {str(e)}]")
            
        return output, type_counter
    
    # Default case for unhandled types
    add_line(f"{type_name}: {str(obj)[:100]}{'...' if len(str(obj)) > 100 else ''}")
    return output, type_counter

def analyze_pkl(file_path):
    """Load and analyze a .pkl file with detailed structure"""
    print(f"\n{'='*50}\nAnalyzing: {file_path}\n{'='*50}")
    
    # Try different loaders
    loaders = [
        ("pickle", lambda f: pickle.load(f)),
        ("joblib", lambda f: joblib.load(f)),
        ("dill", lambda f: __import__('dill').load(f)),
    ]
    
    data = None
    loader_used = None
    
    for name, loader in loaders:
        try:
            with open(file_path, 'rb') as f:
                data = loader(f)
            loader_used = name
            print(f"Successfully loaded using {name}")
            break
        except Exception as e:
            print(f"{name} loader failed: {str(e)}")
    
    if data is None:
        print("\nAll loaders failed. File could not be opened.")
        return
    
    print("\nStructure Summary:")
    print("=" * 60)
    structure, type_counter = explore_structure(data)
    
    for line in structure:
        print(line)
    
    # Print type statistics
    print("\n\nType Distribution:")
    print("=" * 60)
    for typ, count in sorted(type_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{typ}: {count}")
    
    # Show top-level object info
    print("\n\nTop-level Object:")
    print("=" * 60)
    print(f"Type: {get_type_details(data)}")
    if hasattr(data, '__len__'):
        print(f"Length: {len(data)}")
    if hasattr(data, 'shape'):
        print(f"Shape: {data.shape}")
    if hasattr(data, 'dtype'):
        print(f"Dtype: {data.dtype}")

    # Detailed per-key report for dicts
    if isinstance(data, dict):
        print("\nDetailed per-key report:")
        print("=" * 60)
        for k, v in data.items():
            print(f"Key: {repr(k)}")
            print(f"  Type: {get_type_details(v)}")
            if isinstance(v, np.ndarray):
                print(f"  Shape: {v.shape}, Dtype: {v.dtype}, Min: {np.nanmin(v):.4f}, Max: {np.nanmax(v):.4f}")
                print(f"  Sample: {v.flatten()[:5]}")
            elif isinstance(v, (int, float, str, bool, type(None))):
                print(f"  Value: {repr(v)}")
            elif isinstance(v, dict):
                print(f"  Dict with {len(v)} keys. Keys: {list(v.keys())[:5]}")
            elif isinstance(v, (list, tuple)):
                print(f"  {type(v).__name__} of length {len(v)}. Sample: {v[:5]}")
            elif hasattr(v, 'shape') and hasattr(v, 'dtype'):
                print(f"  Shape: {v.shape}, Dtype: {v.dtype}")
            else:
                print(f"  Preview: {str(v)[:100]}")
            print("-")
    
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pkl_inspector.py <path_to_pkl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_pkl(file_path)