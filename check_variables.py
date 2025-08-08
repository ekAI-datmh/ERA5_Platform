import cdsapi
import json

def check_available_variables():
    """
    Check what variables are available in the CDS API for ERA5 datasets.
    """
    try:
        client = cdsapi.Client()
        
        # Check ERA5 single levels dataset
        print("Checking ERA5 single levels dataset...")
        dataset_info = client.dataset('reanalysis-era5-single-levels')
        
        print("\nAvailable variables in reanalysis-era5-single-levels:")
        print("=" * 60)
        
        # Get variables and their details
        variables = dataset_info.get('variables', {})
        
        # Look for temperature-related variables
        temp_vars = []
        for var_name, var_info in variables.items():
            if 'temperature' in var_name.lower() or 'temp' in var_name.lower():
                temp_vars.append(var_name)
                print(f"  {var_name}: {var_info.get('long_name', 'No description')}")
        
        print(f"\nFound {len(temp_vars)} temperature-related variables")
        
        # Also check for some common variables we're trying to use
        target_vars = [
            '2m_temperature', '2m_dewpoint_temperature', 'surface_pressure',
            '10m_u_component_of_wind', '10m_v_component_of_wind',
            '2m_temperature_minimum', '2m_temperature_maximum',
            'surface_net_solar_radiation_sum', 'surface_solar_radiation_downwards_sum',
            'total_precipitation_sum', 'surface_latent_heat_flux',
            'surface_net_solar_radiation', 'surface_sensible_heat_flux',
            'potential_evaporation', 'total_evaporation', 'total_precipitation'
        ]
        
        print("\nChecking our target variables:")
        print("=" * 60)
        available_vars = []
        missing_vars = []
        
        for var in target_vars:
            if var in variables:
                available_vars.append(var)
                print(f"  ✓ {var}: {variables[var].get('long_name', 'Available')}")
            else:
                missing_vars.append(var)
                print(f"  ✗ {var}: NOT FOUND")
        
        print(f"\nSummary:")
        print(f"  Available: {len(available_vars)}/{len(target_vars)}")
        print(f"  Missing: {len(missing_vars)}/{len(target_vars)}")
        
        if missing_vars:
            print(f"\nMissing variables: {missing_vars}")
        
        return available_vars, missing_vars
        
    except Exception as e:
        print(f"Error checking variables: {str(e)}")
        return [], []

if __name__ == "__main__":
    available, missing = check_available_variables() 