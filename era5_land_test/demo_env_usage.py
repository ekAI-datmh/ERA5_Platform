#!/usr/bin/env python3
"""
Demo script showing how to use .env files with CDS API
"""

from era5_data_download import CDSAPIConfig, download_era5_land_data

def demo_env_usage():
    """Demonstrate different ways to use .env files"""
    
    print("=== CDS API .env File Usage Demo ===\n")
    
    # Method 1: Automatic .env loading (looks for .env in current directory)
    print("1. Automatic .env loading:")
    try:
        config = CDSAPIConfig()  # Automatically loads .env file
        print("   ✓ Successfully loaded configuration from .env file")
        print(f"   URL: {config.url}")
        print(f"   Key: {config.key[:10]}..." if config.key else "   Key: Not set")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # Method 2: Specific .env file
    print("2. Specific .env file:")
    try:
        config = CDSAPIConfig(env_file='./custom.env')
        print("   ✓ Successfully loaded configuration from custom.env")
        print(f"   URL: {config.url}")
        print(f"   Key: {config.key[:10]}..." if config.key else "   Key: Not set")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # Method 3: Override .env with direct parameters
    print("3. Override .env with direct parameters:")
    try:
        config = CDSAPIConfig(
            env_file='./env_example.txt',  # Load from example file
            url='https://cds.climate.copernicus.eu/api',  # Override URL
            key='override-key'  # Override key
        )
        print("   ✓ Successfully loaded configuration with overrides")
        print(f"   URL: {config.url}")
        print(f"   Key: {config.key}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

if __name__ == "__main__":
    demo_env_usage() 