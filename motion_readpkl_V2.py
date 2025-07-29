import joblib
import numpy as np
import argparse
from pathlib import Path
import csv

def extract_selected_data(motion_data, keys, csv_path):
    """Extracts selected keys from motion data and saves to CSV."""
    # Create output directory if needed
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect data for selected keys
    extracted = {}
    max_rows = 0
    
    for key in keys:
        if key in motion_data:
            value = motion_data[key]
            
            # Handle different data types
            if np.isscalar(value):
                # Single value - store as a list with one element
                extracted[key] = [value]
                max_rows = max(max_rows, 1)
            elif isinstance(value, np.ndarray):
                # Array data - flatten if needed
                if value.ndim == 1:
                    extracted[key] = value.tolist()
                    max_rows = max(max_rows, len(value))
                elif value.ndim == 2:
                    # Store each column separately
                    for col_idx in range(value.shape[1]):
                        col_key = f"{key}_{col_idx}"
                        extracted[col_key] = value[:, col_idx].tolist()
                    max_rows = max(max_rows, value.shape[0])
                else:
                    # Flatten higher dimensions
                    flat_value = value.reshape(-1)
                    extracted[key] = flat_value.tolist()
                    max_rows = max(max_rows, len(flat_value))
            else:
                # Other data types (list, tuple, etc.)
                try:
                    arr = np.array(value)
                    extracted[key] = arr.reshape(-1).tolist()
                    max_rows = max(max_rows, len(arr))
                except:
                    extracted[key] = [str(value)]
                    max_rows = max(max_rows, 1)
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(extracted.keys())
        
        # Write data row by row
        for i in range(max_rows):
            row = []
            for key in extracted:
                values = extracted[key]
                row.append(values[i] if i < len(values) else '')
            writer.writerow(row)

def inspect_motion_pkl(file_path: Path, output_path: Path = None, keys: list = None):
    """Loads and processes a motion .pkl file."""
    # Load the data
    try:
        data = joblib.load(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Extract the motion data
    motion_key = next(iter(data.keys()))
    motion_data = data[motion_key]
    
    # Process keys if specified
    if keys and output_path:
        extract_selected_data(motion_data, keys, output_path)
    elif output_path:
        # Save full analysis to text file
        with open(output_path, 'w') as f_out:
            f_out.write(f"--- Full Analysis of: {file_path.name} ---\n\n")
            f_out.write(f"Top-level Key: '{motion_key}'\n")
            f_out.write("-" * 40 + "\n")
            
            for key, value in sorted(motion_data.items()):
                f_out.write(f"\n- Key: '{key}'\n")
                if isinstance(value, np.ndarray):
                    f_out.write(f"  Type: numpy.ndarray\n")
                    f_out.write(f"  Shape: {value.shape}\n")
                    f_out.write(f"  Dtype: {value.dtype}\n")
                    np.set_printoptions(threshold=np.inf, linewidth=120)
                    f_out.write(f"  Full data:\n{value}\n")
                else:
                    f_out.write(f"  Type: {type(value).__name__}\n")
                    f_out.write(f"  Value: {repr(value)}\n")
            
            f_out.write("\n--- Analysis Complete ---\n")

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and extract data from motion .pkl files",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('motion_file', type=str, help='Path to the motion .pkl file')
    parser.add_argument('--output', '-o', type=str, help='Output file path (supports .txt or .csv)')
    parser.add_argument('--keys', '-k', type=str, 
                        help='Comma-separated keys to extract (e.g. "dof,fps")\nWhen specified, outputs CSV format')
    
    args = parser.parse_args()
    input_path = Path(args.motion_file)
    output_path = Path(args.output) if args.output else None
    keys = args.keys.split(',') if args.keys else None
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path.resolve()}")
        return
    
    if output_path and keys:
        # Ensure CSV extension when keys are specified
        if output_path.suffix.lower() != '.csv':
            print("Warning: Using CSV format for key extraction. Changing extension to .csv")
            output_path = output_path.with_suffix('.csv')
    
    inspect_motion_pkl(input_path, output_path, keys)
    
    if output_path and keys:
        print(f"Selected keys saved to CSV: {output_path.resolve()}")
    elif output_path:
        print(f"Full analysis saved to text file: {output_path.resolve()}")

if __name__ == "__main__":
    main()