import torch
import os
import sys

def load_pt_file(file_path):
    """Load and inspect .pt file with error handling"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
            
        # Load with CPU compatibility
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
        print(f"\n✅ Successfully loaded: {os.path.basename(file_path)}")
        
        # Inspect contents
        print("\n📦 File contents structure:")
        if isinstance(checkpoint, dict):
            print("Dictionary keys:", list(checkpoint.keys()))
            
            # Special handling for common keys
            if 'state_dict' in checkpoint:
                print("\n🔍 Found 'state_dict' - model weights:")
                print_state_dict_info(checkpoint['state_dict'])
                
            if 'model' in checkpoint:
                print("\n🔍 Found 'model' - full model object")
                
        elif isinstance(checkpoint, torch.nn.Module):
            print("Full model object (nn.Module)")
        else:
            print(f"Object type: {type(checkpoint)}")
            
        return checkpoint
        
    except Exception as e:
        print(f"\n❌ Error loading {file_path}: {str(e)}")
        sys.exit(1)

def print_state_dict_info(state_dict):
    """Print state_dict metadata"""
    print(f"- Keys: {len(state_dict)} parameters")
    print(f"- First 5 keys:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  {i+1}. {key} - Shape: {tuple(state_dict[key].shape)}")
    
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n📊 Total parameters: {total_params:,}")

if __name__ == "__main__":
    # Configure your file path here
    PT_FILE_PATH = "hmr4d_results_messi.pt"  # 🚨 CHANGE THIS TO YOUR ACTUAL PATH
    
    print(f"🔎 Inspecting PyTorch file: {PT_FILE_PATH}")
    checkpoint = load_pt_file(PT_FILE_PATH)
    
    # Additional usage examples:
    print("\n💡 Usage examples:")
    print("1. Access full state_dict:")
    print("   weights = checkpoint['state_dict']")
    
    print("\n2. Load weights into model architecture:")
    print("   model = YourModelClass()")
    print("   model.load_state_dict(weights)")
    
    print("\n3. Extract optimizer state:")
    print("   optimizer = torch.optim.Adam(model.parameters())")
    print("   optimizer.load_state_dict(checkpoint['optimizer'])")