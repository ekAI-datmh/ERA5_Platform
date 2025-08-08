import pygrib
import json
import os
from datetime import datetime

def read_grib_metadata(file_path, output_json_path=None):
    """
    Read GRIB file metadata and store it in a JSON file.
    
    Args:
        file_path (str): Path to the GRIB file
        output_json_path (str): Path for the output JSON file (optional)
    
    Returns:
        dict: Dictionary containing all metadata
    """
    try:
        # Open the GRIB file
        grbs = pygrib.open(file_path)
        
        # Dictionary to store all metadata
        metadata = {
            'file_path': file_path,
            'file_size_bytes': os.path.getsize(file_path),
            'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
            'extraction_time': datetime.now().isoformat(),
            'messages': []
        }
        
        # Iterate through each message in the file
        for grb in grbs:
            message_data = {
                'message_number': grb.messagenumber,
                'edition': grb.edition,
                'metadata': {}
            }
            
            # Collect all available metadata keys and their values
            for key in grb.keys():
                try:
                    value = grb[key]
                    # Convert numpy types to native Python types for JSON serialization
                    if hasattr(value, 'item'):
                        value = value.item()
                    elif hasattr(value, 'tolist'):
                        value = value.tolist()
                    message_data['metadata'][key] = value
                except Exception as e:
                    # Skip keys that can't be accessed
                    message_data['metadata'][key] = f"Error accessing: {str(e)}"
            
            metadata['messages'].append(message_data)
        
        # Close the file
        grbs.close()
        
        # Generate output JSON path if not provided
        if output_json_path is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_json_path = f"{base_name}_metadata.json"
        
        # Write metadata to JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Metadata extracted successfully!")
        print(f"Input file: {file_path}")
        print(f"Output JSON: {output_json_path}")
        print(f"File size: {metadata['file_size_mb']} MB")
        print(f"Number of messages: {len(metadata['messages'])}")
        
        return metadata
        
    except Exception as e:
        print(f"Error reading GRIB file: {e}")
        return None

def read_grib_metadata_batch(input_dir, output_dir=None):
    """
    Read metadata from multiple GRIB files in a directory.
    
    Args:
        input_dir (str): Directory containing GRIB files
        output_dir (str): Directory for output JSON files (optional)
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all GRIB files
    grib_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.grib') or file.endswith('.grb'):
                grib_files.append(os.path.join(root, file))
    
    print(f"Found {len(grib_files)} GRIB files to process")
    
    # Process each file
    for i, grib_file in enumerate(grib_files, 1):
        print(f"\nProcessing file {i}/{len(grib_files)}: {os.path.basename(grib_file)}")
        
        # Generate output JSON path
        base_name = os.path.splitext(os.path.basename(grib_file))[0]
        output_json = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Extract metadata
        metadata = read_grib_metadata(grib_file, output_json)
        
        if metadata:
            print(f"✓ Successfully processed: {os.path.basename(grib_file)}")
        else:
            print(f"✗ Failed to process: {os.path.basename(grib_file)}")

# Usage examples
if __name__ == "__main__":
    # Example 1: Process a single file
    file_path = '/media/ekai2/data2tb/datmh/Platform/data/era5_vietnam/grib/2023/01/000000000000000000a4_2023_01.grib'
    
    if os.path.exists(file_path):
        print("Processing single file...")
        metadata = read_grib_metadata(file_path)
    else:
        print(f"File not found: {file_path}")
        print("Trying to find GRIB files in the data directory...")
        
        # Example 2: Process all GRIB files in a directory
        data_dir = '/media/ekai2/data2tb/datmh/Platform/data/era5_vietnam/grib'
        if os.path.exists(data_dir):
            print(f"Processing all GRIB files in: {data_dir}")
            read_grib_metadata_batch(data_dir)
        else:
            print(f"Data directory not found: {data_dir}")
            print("Please update the file_path variable with a valid GRIB file path.")