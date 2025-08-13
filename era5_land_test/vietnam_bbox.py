#!/usr/bin/env python3
"""
Vietnam Bounding Box Coordinates for ERA5 Data Download
"""

# Vietnam bounding box coordinates
# Format: [north, west, south, east]
VIETNAM_BBOX = [23.5, 102.0, 8.0, 110.0]

# Alternative: More precise Vietnam bounding box
VIETNAM_BBOX_PRECISE = [23.5, 102.0, 8.0, 110.0]

# Regional bounding boxes for different parts of Vietnam
NORTH_VIETNAM_BBOX = [23.5, 102.0, 15.0, 110.0]  # Northern region
CENTRAL_VIETNAM_BBOX = [15.0, 102.0, 10.0, 110.0]  # Central region
SOUTH_VIETNAM_BBOX = [10.0, 102.0, 8.0, 110.0]   # Southern region

# Major cities in Vietnam with their coordinates
VIETNAM_CITIES = {
    'Hanoi': {'lat': 21.0285, 'lon': 105.8542},
    'Ho Chi Minh City': {'lat': 10.8231, 'lon': 106.6297},
    'Da Nang': {'lat': 16.0544, 'lon': 108.2022},
    'Hai Phong': {'lat': 20.8449, 'lon': 106.6881},
    'Can Tho': {'lat': 10.0452, 'lon': 105.7469},
    'Hue': {'lat': 16.4637, 'lon': 107.5909},
    'Nha Trang': {'lat': 12.2388, 'lon': 109.1967},
    'Vung Tau': {'lat': 10.3459, 'lon': 107.0843},
    'Phu Quoc': {'lat': 10.2233, 'lon': 103.9589},
    'Sapa': {'lat': 22.3364, 'lon': 103.8440},
}

def get_vietnam_bbox(region='full', buffer_degrees=0.5):
    """
    Get Vietnam bounding box coordinates
    
    Args:
        region (str): 'full', 'north', 'central', 'south', or 'precise'
        buffer_degrees (float): Additional buffer around the bounding box
    
    Returns:
        list: [north, west, south, east] coordinates
    """
    bbox_map = {
        'full': VIETNAM_BBOX,
        'precise': VIETNAM_BBOX_PRECISE,
        'north': NORTH_VIETNAM_BBOX,
        'central': CENTRAL_VIETNAM_BBOX,
        'south': SOUTH_VIETNAM_BBOX,
    }
    
    if region not in bbox_map:
        raise ValueError(f"Region must be one of: {list(bbox_map.keys())}")
    
    bbox = bbox_map[region].copy()
    
    # Add buffer if specified
    if buffer_degrees > 0:
        bbox[0] += buffer_degrees  # north
        bbox[1] -= buffer_degrees  # west
        bbox[2] -= buffer_degrees  # south
        bbox[3] += buffer_degrees  # east
    
    return bbox

def print_vietnam_info():
    """Print comprehensive information about Vietnam coordinates"""
    print("=== Vietnam Bounding Box Coordinates ===\n")
    
    print("Full Vietnam Bounding Box:")
    print(f"  [north, west, south, east] = {VIETNAM_BBOX}")
    print(f"  North: {VIETNAM_BBOX[0]}°N")
    print(f"  West:  {VIETNAM_BBOX[1]}°E")
    print(f"  South: {VIETNAM_BBOX[2]}°N")
    print(f"  East:  {VIETNAM_BBOX[3]}°E")
    print()
    
    print("Regional Bounding Boxes:")
    print(f"  North Vietnam:  {NORTH_VIETNAM_BBOX}")
    print(f"  Central Vietnam: {CENTRAL_VIETNAM_BBOX}")
    print(f"  South Vietnam:  {SOUTH_VIETNAM_BBOX}")
    print()
    
    print("Major Cities Coordinates:")
    for city, coords in VIETNAM_CITIES.items():
        print(f"  {city}: {coords['lat']}°N, {coords['lon']}°E")
    print()
    
    print("Usage with ERA5 Data Download:")
    print("```python")
    print("from era5_data_download import download_era5_land_data")
    print("from vietnam_bbox import get_vietnam_bbox")
    print()
    print("# Download data for all of Vietnam")
    print("vietnam_area = get_vietnam_bbox('full')")
    print("success = download_era5_land_data(")
    print("    variable='2m_temperature',")
    print("    year=2023,")
    print("    month=1,")
    print("    day=1,")
    print("    time=['00:00', '06:00', '12:00', '18:00'],")
    print("    area=vietnam_area,")
    print("    output_file='vietnam_temp_20230101.nc'")
    print(")")
    print("```")

def demo_vietnam_download():
    """Demonstrate downloading ERA5 data for Vietnam"""
    try:
        from era5_data_download import download_era5_land_data
        
        print("=== Vietnam ERA5 Data Download Demo ===\n")
        
        # Get Vietnam bounding box
        vietnam_area = get_vietnam_bbox('full')
        print(f"Vietnam bounding box: {vietnam_area}")
        
        # Download example (commented out to avoid actual download)
        print("\nExample download command (commented out):")
        print("# success = download_era5_land_data(")
        print("#     variable='2m_temperature',")
        print("#     year=2023,")
        print("#     month=1,")
        print("#     day=1,")
        print("#     time=['00:00', '06:00', '12:00', '18:00'],")
        print("#     area=vietnam_area,")
        print("#     output_file='vietnam_temp_20230101.nc'")
        print("# )")
        
        print("\nTo actually download data, uncomment the lines above and run the script.")
        
    except ImportError:
        print("Error: era5_data_download module not found.")
        print("Make sure you have the era5_data_download.py file in the same directory.")

if __name__ == "__main__":
    print_vietnam_info()
    print("\n" + "="*50 + "\n")
    demo_vietnam_download() 