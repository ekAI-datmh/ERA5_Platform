import xarray as xr
import rioxarray # Enables the .rio accessor for geospatial operations
from pathlib import Path
import os
import numpy as np # For datetime handling
from datetime import datetime, timedelta # For date calculations
import warnings
from rasterio.enums import Resampling # For reproject_match
import rasterio
from typing import Optional # Added for Optional type hint

# --- Global Configuration ---

# This mapping is crucial for aligning variables from CDS NetCDF files (which use short names)
# with the GEE GeoTIFF band descriptions (which use long names).
CDS_SHORT_TO_LONG_NAME_MAP = {
    'd2m': '2m_dewpoint_temperature',
    't2m': '2m_temperature',
    'skt': 'skin_temperature',
    'slhf': 'surface_latent_heat_flux',
    'ssr': 'surface_net_solar_radiation',
    'sshf': 'surface_sensible_heat_flux',
    'ssrd': 'surface_solar_radiation_downwards',
    'pev': 'potential_evaporation',
    'e': 'total_evaporation',
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'sp': 'surface_pressure',
    'tp': 'total_precipitation',
}

class ERA5LandProcessor:
    """
    A robust class for processing ERA5-Land NetCDF data.
    It handles intelligent file selection, preprocessing, slicing, clipping,
    and saving hourly multi-band GeoTIFFs from daily NetCDF files.
    """

    def __init__(self, base_data_dir: Path, output_base_dir: Path, timezone_offset_hours: int = 0):
        """
        Initializes the processor with paths to input data and output directories.

        Args:
            base_data_dir (Path): The root directory where your daily NetCDF files are stored
                                  (e.g., 'data' if files are in 'data/YYYY/MM/').
            output_base_dir (Path): The root directory where processed GeoTIFFs will be saved.
        """
        if not base_data_dir.is_dir():
            warnings.warn(f"Base data directory not found: {base_data_dir}. This is okay if you only plan to use the comparison function.")
        self.base_data_dir = base_data_dir
        self.output_base_dir = output_base_dir
        self.output_base_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        self.timezone_offset_hours = timezone_offset_hours

    def _preprocess_era5_single_file(self, ds: xr.Dataset) -> xr.Dataset:
        # Rename 'valid_time' to 'time' for consistency
        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        
        # Handle 'expver' dimension (common in ERA5, often has a single value 1)
        if 'expver' in ds.coords: # Check if 'expver' coordinate exists
            # Only select expver=1 if it's a dimension with multiple values
            if 'expver' in ds.dims and len(ds['expver']) > 1:
                ds = ds.sel(expver=1)
            # Always drop the 'expver' coordinate/dimension after handling (if it exists)
            ds = ds.drop_vars('expver', errors='ignore') 
            
        # Ensure CRS is set early on the dataset
        if ds.rio.crs is None:
            ds.rio.write_crs("EPSG:4326", inplace=True) # ERA5 is WGS84
        
        # Explicitly set spatial dimensions here. This is important for context propagation.
        if 'latitude' in ds.dims and 'longitude' in ds.dims:
            ds = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude', inplace=True)

        return ds

    def _get_relevant_nc_filepath(self, date_dt: datetime, base_dir: Path) -> Path:
        """
        Constructs the expected file path for a daily NetCDF file for a given date.
        
        Args:
            date_dt (datetime): The date for which to find the file.
            base_dir (Path): The base data directory to search within.

        Returns:
            Path: The constructed Path object for the NetCDF file.
        """
        year = date_dt.year
        month = date_dt.month
        
        # Filename is based on the local date, as set by the crawler
        file_path = base_dir / str(year) / f"{month:02d}" / f"era5_vietnam_{date_dt.strftime('%Y_%m_%d')}.nc"
        return file_path

    def _get_cumulative_gee_precipitation(self, comparison_dt: datetime, gee_data_dir: Path) -> Optional[xr.DataArray]:
        """
        Calculates the cumulative total precipitation from GEE hourly GeoTIFFs up to the comparison hour.
        GEE provides 'total_precipitation_hourly' as incremental, while CDS has cumulative 'total_precipitation'.
        """
        print(f"  Calculating cumulative GEE precipitation for {comparison_dt.isoformat()}...")
        
        # Define the local date for which we need to sum hourly files
        local_date = comparison_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        month_output_dir = gee_data_dir / str(local_date.year) / f"{local_date.month:02d}"

        cumulative_precip_data = None
        target_shape = None
        target_transform = None
        target_crs = None

        # Iterate from 00:00 up to the comparison_dt.hour (inclusive)
        for h in range(comparison_dt.hour + 1):
            current_hour_dt = local_date + timedelta(hours=h)
            gee_tif_filename = f"era5_vietnam_GEE_{current_hour_dt.strftime('%Y%m%dT%H%M')}.tif"
            gee_tif_path = month_output_dir / gee_tif_filename

            if not gee_tif_path.exists():
                print(f"    GEE hourly precipitation file not found: {gee_tif_path}. Cannot calculate cumulative precipitation.")
                return None

            try:
                with rioxarray.open_rasterio(gee_tif_path, masked=True) as ds_hourly_gee:
                    # Find the correct precipitation band (total_precipitation_hourly)
                    # Assuming it's in the band descriptions if combined correctly
                    tp_hourly_gee_da = None
                    if 'variable' in ds_hourly_gee.dims: # Check if bands were renamed to 'variable'
                        if 'total_precipitation_hourly' in ds_hourly_gee['variable'].values:
                            tp_hourly_gee_da = ds_hourly_gee.sel(variable='total_precipitation_hourly')
                    else: # Fallback if bands were not renamed, try to find by description or band index
                        # This is a bit fragile, relies on naming conventions or knowledge of band order
                        # For now, assume it's correctly named 'total_precipitation_hourly' in a band description
                        for band_idx, desc in enumerate(ds_hourly_gee.attrs.get('descriptions', [])):
                            if desc == 'total_precipitation_hourly':
                                tp_hourly_gee_da = ds_hourly_gee.isel(band=band_idx) # Select by index if no 'variable' dim
                                break
                        if tp_hourly_gee_da is None and 'total_precipitation_hourly' in ds_hourly_gee.data_vars:
                             tp_hourly_gee_da = ds_hourly_gee['total_precipitation_hourly']

                    if tp_hourly_gee_da is None:
                        print(f"    'total_precipitation_hourly' band not found in {gee_tif_path}. Skipping this hour for accumulation.")
                        continue
                    
                    # Ensure the DataArray has spatial dims and CRS
                    if tp_hourly_gee_da.rio.crs is None:
                        tp_hourly_gee_da = tp_hourly_gee_da.rio.write_crs("EPSG:4326")
                    if 'y' in tp_hourly_gee_da.dims and 'x' in tp_hourly_gee_da.dims:
                        tp_hourly_gee_da = tp_hourly_gee_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
                    else:
                         print(f"    Warning: GEE precipitation data missing y/x spatial dimensions for {gee_tif_path.name}. Reprojection may fail.")

                    # For the first hour, initialize cumulative_precip_data
                    if cumulative_precip_data is None:
                        cumulative_precip_data = tp_hourly_gee_da.copy()
                        target_shape = tp_hourly_gee_da.shape
                        target_transform = tp_hourly_gee_da.rio.transform()
                        target_crs = tp_hourly_gee_da.rio.crs
                    else:
                        # Ensure current hourly data matches the grid of the cumulative data
                        # Use reproject_match to align them before summing
                        if not (tp_hourly_gee_da.rio.crs == target_crs and 
                                tp_hourly_gee_da.rio.transform() == target_transform and
                                tp_hourly_gee_da.shape == target_shape):
                            # print(f"    Aligning {gee_tif_path.name} to cumulative grid for summation.")
                            tp_hourly_gee_da_aligned = tp_hourly_gee_da.rio.reproject_match(
                                xr.DataArray(np.zeros(target_shape), 
                                             coords={'y': cumulative_precip_data.y, 'x': cumulative_precip_data.x}, 
                                             dims=['y','x']).rio.write_crs(target_crs),
                                resampling=Resampling.nearest, # Nearest is fine for sums of discrete values
                                nodata=np.nan
                            )
                            cumulative_precip_data += tp_hourly_gee_da_aligned
                        else:
                            cumulative_precip_data += tp_hourly_gee_da

            except Exception as e:
                print(f"    Error processing GEE hourly precipitation file {gee_tif_path}: {e}")
                return None
        
        if cumulative_precip_data is not None: 
            # Ensure the final cumulative data array has correct spatial context
            cumulative_precip_data = cumulative_precip_data.rio.set_spatial_dims(x_dim='x', y_dim='y')
            cumulative_precip_data = cumulative_precip_data.rio.write_crs(target_crs)
        
        return cumulative_precip_data

    def process_time_series_to_geotiffs(
        self,
        start_timestamp: str,
        end_timestamp: str,
        bbox: list, # [min_lon, min_lat, max_lon, max_lat]
        variables_to_extract: list # Short names like ['t2m', 'tp']
    ) -> list[Path]:
        
        # print(f"\n--- Starting Time Series GeoTIFF Processing ---")
        # print(f"Time range: {start_timestamp} to {end_timestamp}")
        # print(f"Target BBox: {bbox}")
        # print(f"Variables for output: {variables_to_extract}")
        
        saved_files = []
        
        try:
            start_dt = datetime.fromisoformat(start_timestamp)
            end_dt = datetime.fromisoformat(end_timestamp)
            
            # Generate all individual hourly datetime objects within the range
            current_hour_dt = start_dt
            all_hours_to_process = []
            while current_hour_dt <= end_dt:
                all_hours_to_process.append(current_hour_dt)
                current_hour_dt += timedelta(hours=1)
            
            if not all_hours_to_process:
                print("No hourly slices identified in the specified time range. Exiting.")
                return saved_files

            # print(f"\nIdentified {len(all_hours_to_process)} hourly slices to process.")
            
            # --- Iterate through each desired output hour ---
            for i, hourly_dt in enumerate(all_hours_to_process):
                # Determine the specific daily file that contains this hour
                daily_nc_path = self._get_relevant_nc_filepath(hourly_dt, self.base_data_dir)
                
                if not daily_nc_path.exists():
                    print(f"Skipping: Daily file for {hourly_dt.date()} not found: {daily_nc_path}")
                    continue

                    # print(f"Processing hour {hourly_dt.isoformat()}: Loading {daily_nc_path.name}...")
                    
                try:
                    # Open only the relevant daily file
                    with xr.open_dataset(daily_nc_path, engine='netcdf4') as ds_daily:
                        # Apply preprocessing to standardize the daily dataset
                        ds_daily = self._preprocess_era5_single_file(ds_daily)
                        
                        # Select the specific hour from this daily dataset
                        # .sel() with method='nearest' is robust if exact timestamp isn't there
                        hourly_data_slice_ds = ds_daily.sel(time=hourly_dt, method='nearest')
                        
                        # Select only the requested variables (bands) for this hourly slice
                        # This creates a new Dataset containing only the desired bands
                        ds_for_tiff = hourly_data_slice_ds[variables_to_extract]

                        # Clip the data to the bounding box
                        min_lon, min_lat, max_lon, max_lat = bbox
                        clipped_data = ds_for_tiff.rio.clip_box(
                            minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat
                        )
                        
                        # Construct output filename (era5_vietnam_YYYYMMDDTHHMM.tif)
                        timestamp_str = hourly_dt.strftime('%Y_%m_%d_T%H') # e.g., 2024_01_01_T12
                        tif_filename = self.output_base_dir / f'era5_vietnam_{timestamp_str}.tif'

                        # Save as multi-band GeoTIFF
                        # rioxarray automatically maps data variables to TIFF bands
                        clipped_data.rio.to_raster(tif_filename, tiled=True, compress='LZW', num_threads='all_cpus')
                        
                        saved_files.append(tif_filename)
                    
                        
                except Exception as e:
                    print(f"Error processing hour {hourly_dt.isoformat()} from {daily_nc_path.name}: {e}")
                    continue 

            print("\n--- All Hourly Slices Processed ---")
            print(f"Total GeoTIFFs saved: {len(saved_files)}")
            
        except Exception as e:
            print(f"An unexpected error occurred during the overall processing setup: {e}")
        
        return saved_files

    def compare_cds_vs_gee_at_hour(self, comparison_dt: datetime, cds_data_dir: Path, gee_data_dir: Path):
        """
        Compares data from a CDS NetCDF file and a GEE GeoTIFF file for a specific hour.

        Args:
            comparison_dt (datetime): The local datetime for the hour to compare.
            cds_data_dir (Path): The root directory of the CDS NetCDF data.
            gee_data_dir (Path): The root directory of the GEE GeoTIFF data.

        Returns:
            dict: A dictionary containing MAE and RMSE for each common variable.
        """
        print(f"\n--- Comparing CDS vs GEE for hour: {comparison_dt.strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        # --- 1. Load CDS Data Slice for the Hour ---
        cds_nc_path = self._get_relevant_nc_filepath(comparison_dt, cds_data_dir)
        if not cds_nc_path.exists():
            print(f"  ❌ Error: CDS file not found: {cds_nc_path}")
            return None
        
        try:
            # Convert comparison_dt (local time) to UTC for CDS data selection
            utc_comparison_dt = comparison_dt - timedelta(hours=self.timezone_offset_hours)
            
            with xr.open_dataset(cds_nc_path) as ds_daily_cds:
                ds_daily_cds = self._preprocess_era5_single_file(ds_daily_cds)
                
                # Debugging: Print time coordinate info
                print(f"  CDS File Time Coords (first 5): {ds_daily_cds.time.values.flatten()[:5]}")
                print(f"  CDS Time Coords Dtype: {ds_daily_cds.time.dtype}")
                print(f"  Target UTC for CDS selection: {utc_comparison_dt.isoformat()}")

                # Select the exact hour from the CDS daily file using UTC datetime
                cds_hourly_slice = ds_daily_cds.sel(time=utc_comparison_dt, method='nearest')
            
            # --- Ensure cds_hourly_slice has correct spatial dims and CRS just after slicing ---
            # This is crucial as slicing might sometimes drop rioxarray context.
            if 'latitude' in cds_hourly_slice.dims and 'longitude' in cds_hourly_slice.dims:
                if cds_hourly_slice.rio.crs is None:
                    cds_hourly_slice = cds_hourly_slice.rio.write_crs("EPSG:4326")
                cds_hourly_slice = cds_hourly_slice.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
            else:
                print(f"    ⚠️ Warning: CDS hourly slice missing spatial dimensions. Reprojection may fail.")

            # Debugging: Print selected slice info
            print(f"  Selected CDS slice has dimensions: {cds_hourly_slice.sizes}")
            # Check a few key variables in the selected slice
            for var_name in ['t2m', 'tp', 'd2m', 'skt', 'slhf', 'ssr', 'sshf', 'ssrd', 'sp', 'u10', 'v10', 'pev', 'e']:
                if var_name in cds_hourly_slice.data_vars:
                    val = cds_hourly_slice[var_name].values.item() if cds_hourly_slice[var_name].ndim == 0 else cds_hourly_slice[var_name].values.flatten()[0] # Get a single value or first if multi-dim
                    print(f"  CDS slice '{var_name}' value at selected time (first pixel): {val}")
                else:
                    print(f"  CDS slice does not contain variable: {var_name}")

            print(f"  ✅ Loaded CDS data from: {cds_nc_path.name}")
        except Exception as e:
            print(f"  ❌ Error loading or slicing CDS file {cds_nc_path.name}: {e}")
            return None

        # --- 2. Load GEE Data for the Hour ---
        gee_tif_filename = f"era5_vietnam_GEE_{comparison_dt.strftime('%Y%m%dT%H%M')}.tif"
        gee_tif_path = gee_data_dir / str(comparison_dt.year) / f"{comparison_dt.month:02d}" / gee_tif_filename
        if not gee_tif_path.exists():
            print(f"  ❌ Error: GEE file not found: {gee_tif_path}")
            return None
            
        try:
            # More robust way to read band descriptions:
            # First, open with rasterio just to get metadata
            with rasterio.open(gee_tif_path) as src:
                band_descriptions = src.descriptions

            # Then, open the data with rioxarray
            gee_ds = rioxarray.open_rasterio(gee_tif_path, masked=True)

            # Assign descriptions to the 'band' coordinate first
            # Ensure band_descriptions is a list for assignment
            if band_descriptions and len(band_descriptions) == len(gee_ds.coords['band']):
                # Directly assign values to the coordinate's underlying NumPy array
                gee_ds = gee_ds.assign_coords(band=np.array(band_descriptions, dtype=object))
                # Then rename the dimension and its coordinate
                gee_ds = gee_ds.rename({'band': 'variable'})
            else:
                # If no descriptions or mismatch, fallback to default band names
                print(f"  Warning: Band descriptions not found or length mismatch in GEE GeoTIFF for {gee_tif_path.name}. Bands might be named 'band_data'.")
                # Create generic variable names based on band index if renaming failed
                gee_ds = gee_ds.assign_coords(band=[f"band_{i+1}" for i in range(len(gee_ds.coords['band']))])
                gee_ds = gee_ds.rename({'band': 'variable'})

            print(f"  ✅ Loaded GEE data from: {gee_tif_path.name}")
        except Exception as e:
            print(f"  ❌ Error loading GEE file {gee_tif_path.name}: {e}")
            return None

        # --- 3. Align Grids and Compare Variables ---
        results = {}
        # Iterate through variables found in the CDS NetCDF slice
        for var_short_name in cds_hourly_slice.data_vars:
            var_short_name_str = str(var_short_name)

            # Skip total_precipitation (tp) as it's handled separately
            if var_short_name_str == 'tp':
                continue
            
            var_long_name = CDS_SHORT_TO_LONG_NAME_MAP.get(var_short_name_str)
            
            if not var_long_name or var_long_name not in gee_ds['variable'].values:
                print(f"  - Skipping variable '{var_short_name_str}': Not found in GEE file or mapping.")
                continue
            
            print(f"\n  --- Comparing variable: {var_short_name_str} ({var_long_name}) ---")
            
            # --- CRITICAL FIX: Robustly create cds_data_var with explicit spatial context ---
            # Directly construct the DataArray for the variable, ensuring rioxarray context is present.
            cds_raw_data = cds_hourly_slice[var_short_name_str].values
            cds_latitude = cds_hourly_slice['latitude'].values
            cds_longitude = cds_hourly_slice['longitude'].values

            # Create a new DataArray with explicit dimensions and coordinates
            cds_data_var = xr.DataArray(
                cds_raw_data,
                coords={
                    'latitude': cds_latitude,
                    'longitude': cds_longitude
                },
                dims=['latitude', 'longitude']
            )
            # Now set spatial dimensions and CRS explicitly using rioxarray accessor
            cds_data_var = cds_data_var.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
            cds_data_var = cds_data_var.rio.write_crs("EPSG:4326") # ERA5 is WGS84

            # Original GEE data variable
            gee_data_var = gee_ds.sel(variable=var_long_name)
            
            # --- Debugging prints for CRS, Dims, and Grid Info before reprojection ---
            print(f"    CDS data_var CRS before reproject: {cds_data_var.rio.crs}")
            print(f"    CDS data_var Dims before reproject: {cds_data_var.dims}")
            print(f"    CDS data_var Transform: {cds_data_var.rio.transform()}")
            print(f"    CDS data_var Resolution: {cds_data_var.rio.resolution()}")
            print(f"    CDS data_var Bounds: {cds_data_var.rio.bounds()}")
            print(f"    CDS data_var Shape: {cds_data_var.shape}")
            print(f"    CDS data_var Non-NaN count before reproject: {np.count_nonzero(~np.isnan(cds_data_var.values))}")

            print(f"    GEE data_var CRS before reproject: {gee_data_var.rio.crs}")
            print(f"    GEE data_var Dims before reproject: {gee_data_var.dims}")
            print(f"    GEE data_var Transform: {gee_data_var.rio.transform()}")
            print(f"    GEE data_var Resolution: {gee_data_var.rio.resolution()}")
            print(f"    GEE data_var Bounds: {gee_data_var.rio.bounds()}")
            print(f"    GEE data_var Shape: {gee_data_var.shape}")
            print(f"    GEE data_var Non-NaN count before reproject: {np.count_nonzero(~np.isnan(gee_data_var.values))}")

            # Align CDS data grid to match GEE grid using rioxarray.reproject_match.
            # This is more robust than simple interpolation as it handles differences
            # in CRS, grid resolution, and alignment (affine transform).
            print("    Aligning CDS grid to match GEE grid using rioxarray.reproject_match...")
            cds_aligned = cds_data_var.rio.reproject_match(
                gee_data_var,
                resampling=Resampling.bilinear,  # Use bilinear for continuous data like temperature
                nodata=np.nan
            )

            print(f"    CDS aligned CRS after reproject_match: {cds_aligned.rio.crs}")
            print(f"    CDS aligned Dims after reproject_match: {cds_aligned.dims}")
            print(f"    CDS aligned Non-NaN count after reproject_match: {np.count_nonzero(~np.isnan(cds_aligned.values))}")

            # Get numpy arrays for comparison
            cds_vals = cds_aligned.values
            gee_vals = gee_data_var.values.squeeze() # Squeeze to remove single band dimension if present

            # --- Apply Unit Conversions if necessary (after alignment and before comparison/filtering) ---
            # ERA5-Land temperature (t2m, d2m, skt) is often in Kelvin (K).
            # If GEE data is also in Kelvin, convert both to Celsius for more interpretable comparison.
            if var_short_name_str in ['t2m', 'd2m', 'skt']:
                print("    Converting temperature to Celsius (K to C)...")
                cds_vals = cds_vals - 273.15
                gee_vals = gee_vals - 273.15

            # --- Debugging: Print sample values (after conversions) ---
            print(f"    Sample CDS values (first 5): {cds_vals.flatten()[:5]}")
            print(f"    Sample GEE values (first 5): {gee_vals.flatten()[:5]}")

            # --- Pre-check CDS data validity ---
            if np.all(np.isnan(cds_vals)) or cds_vals.size == 0:
                print(f"    ⚠️ Warning: CDS data for '{var_short_name_str}' is all NaN or empty. Skipping comparison for this variable.")
                results[var_short_name_str] = {'mae': None, 'rmse': None, 'compared_pixels': 0, 'status': 'CDS_DATA_INVALID'}
                continue

            # Create a mask to filter values within the specified range and ignore NaNs
            # Adjust the range based on the units and expected physical values for each variable.
            comparison_min = -np.inf # Default to no lower bound
            comparison_max = np.inf  # Default to no upper bound

            if var_short_name_str in ['t2m', 'd2m', 'skt']:
                comparison_min = -50  # Celsius
                comparison_max = 50   # Celsius
            elif var_short_name_str in ['slhf', 'sshf']:
                # Temporarily remove bounds to debug pixel count issue
                comparison_min = -np.inf 
                comparison_max = np.inf
                print(f"    WARNING: Temporarily removed value range filter for {var_short_name_str} to debug pixel count.")
            elif var_short_name_str in ['ssr', 'ssrd']:
                comparison_min = 0     # J/m^2 (solar radiation is non-negative)
                comparison_max = 2_000_000 # Max typical hourly accumulated solar radiation
            elif var_short_name_str == 'sp': # Surface pressure in Pascals
                comparison_min = 50_000 # Typical min pressure
                comparison_max = 110_000 # Typical max pressure
            elif var_short_name_str in ['u10', 'v10']:
                comparison_min = -50 # m/s (wind components)
                comparison_max = 50 # m/s
            elif var_short_name_str in ['pev', 'e']:
                comparison_min = -0.001 # meters (accumulated evaporation - can be slightly negative due to precision)
                comparison_max = 0.1 # 100mm per hour - very high, but covers extremes
            # Note: total_precipitation (tp) is handled separately due to accumulation differences

            # Add more detailed NaN and validity checks before masking
            print(f"    CDS non-NaN count before value filter: {np.count_nonzero(~np.isnan(cds_vals))}")
            print(f"    GEE non-NaN count before value filter: {np.count_nonzero(~np.isnan(gee_vals))}")

            # --- Debugging: Print sample values for pev, e, slhf, sshf if relevant ---
            if var_short_name_str in ['pev', 'e', 'slhf', 'sshf']:
                print(f"    Sample CDS filtered values ({cds_filtered.size} pixels): {cds_filtered.flatten()[:20]}")
                print(f"    Sample GEE filtered values ({gee_filtered.size} pixels): {gee_filtered.flatten()[:20]}")

            mask = (
                (cds_vals >= comparison_min) & (cds_vals <= comparison_max) &
                (gee_vals >= comparison_min) & (gee_vals <= comparison_max) &
                (~np.isnan(cds_vals)) & (~np.isnan(gee_vals))
            )
            
            cds_filtered = cds_vals[mask]
            gee_filtered = gee_vals[mask]

            print(f"    CDS filtered non-NaN count: {cds_filtered.size}")
            print(f"    GEE filtered non-NaN count: {gee_filtered.size}")

            # Add a check for all NaNs after filtering as well
            if np.all(np.isnan(cds_filtered)) or cds_filtered.size == 0:
                print(f"    ⚠️ Warning: No valid overlapping pixels found after value range filtering for '{var_short_name_str}'. Skipping comparison.")
                results[var_short_name_str] = {'mae': None, 'rmse': None, 'compared_pixels': 0, 'status': 'NO_COMPARABLE_PIXELS_AFTER_FILTERING'}
                continue
            
            # Calculate metrics
            mae = np.mean(np.abs(cds_filtered - gee_filtered))
            rmse = np.sqrt(np.mean((cds_filtered - gee_filtered)**2))
            
            results[var_short_name_str] = {'mae': mae, 'rmse': rmse, 'compared_pixels': cds_filtered.size, 'status': 'COMPLETED'}
            
            print(f"    Compared Pixels: {cds_filtered.size}")
            print(f"    MAE: {mae:.4f}")
            print(f"    RMSE: {rmse:.4f}")

        # --- Special handling for total_precipitation (tp) ---
        # This variable requires cumulative summation for GEE data
        var_short_name_tp = 'tp'
        var_long_name_tp = CDS_SHORT_TO_LONG_NAME_MAP.get(var_short_name_tp)

        if var_long_name_tp and var_short_name_tp in cds_hourly_slice.data_vars:
            print(f"\n  --- Comparing variable: {var_short_name_tp} (Cumulative from GEE) ---")
            cds_tp_data_var = cds_hourly_slice[var_short_name_tp]
            cds_tp_data_var = cds_tp_data_var.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
            cds_tp_data_var = cds_tp_data_var.rio.write_crs("EPSG:4326")

            # Get cumulative GEE precipitation up to the comparison hour
            cumulative_gee_tp_da = self._get_cumulative_gee_precipitation(comparison_dt, gee_data_dir)

            if cumulative_gee_tp_da is None:
                print(f"    ❌ Error: Could not get cumulative GEE precipitation for {comparison_dt.isoformat()}. Skipping comparison for '{var_short_name_tp}'.")
                results[var_short_name_tp] = {'mae': None, 'rmse': None, 'compared_pixels': 0, 'status': 'GEE_TP_ACCUMULATION_FAILED'}
            else:
                # Align CDS TP data to the GEE cumulative TP grid
                cds_tp_aligned = cds_tp_data_var.rio.reproject_match(
                    cumulative_gee_tp_da,
                    resampling=Resampling.bilinear, 
                    nodata=np.nan
                )

                cds_tp_vals = cds_tp_aligned.values
                gee_tp_vals = cumulative_gee_tp_da.values.squeeze()

                # Debugging prints for tp
                print(f"    CDS tp non-NaN count before reproject: {np.count_nonzero(~np.isnan(cds_tp_data_var.values))}")
                print(f"    GEE tp non-NaN count before reproject: {np.count_nonzero(~np.isnan(cumulative_gee_tp_da.values))}")
                print(f"    CDS tp aligned non-NaN count: {np.count_nonzero(~np.isnan(cds_tp_aligned.values))}")
                print(f"    Sample CDS tp values (first 5): {cds_tp_vals.flatten()[:5]}")
                print(f"    Sample GEE tp values (first 5): {gee_tp_vals.flatten()[:5]}")

                # Apply value range filter for precipitation (meters)
                tp_comparison_min = -0.001 # Allow slightly negative due to precision
                tp_comparison_max = 0.1 # 100mm per hour

                tp_mask = (
                    (cds_tp_vals >= tp_comparison_min) & (cds_tp_vals <= tp_comparison_max) &
                    (gee_tp_vals >= tp_comparison_min) & (gee_tp_vals <= tp_comparison_max) &
                    (~np.isnan(cds_tp_vals)) & (~np.isnan(gee_tp_vals))
                )

                cds_tp_filtered = cds_tp_vals[tp_mask]
                gee_tp_filtered = gee_tp_vals[tp_mask]

                print(f"    CDS tp filtered non-NaN count: {cds_tp_filtered.size}")
                print(f"    GEE tp filtered non-NaN count: {gee_tp_filtered.size}")

                if cds_tp_filtered.size == 0:
                    print(f"    ⚠️ Warning: No valid overlapping pixels found for 'tp' after value range filtering. Skipping comparison.")
                    results[var_short_name_tp] = {'mae': None, 'rmse': None, 'compared_pixels': 0, 'status': 'NO_COMPARABLE_TP_PIXELS_AFTER_FILTERING'}
                else:
                    mae_tp = np.mean(np.abs(cds_tp_filtered - gee_tp_filtered))
                    rmse_tp = np.sqrt(np.mean((cds_tp_filtered - gee_tp_filtered)**2))
                    results[var_short_name_tp] = {'mae': mae_tp, 'rmse': rmse_tp, 'compared_pixels': cds_tp_filtered.size, 'status': 'COMPLETED'}

                    print(f"    Compared Pixels: {cds_tp_filtered.size}")
                    print(f"    MAE: {mae_tp:.4f}")
                    print(f"    RMSE: {rmse_tp:.4f}")

        return results

# --- Example Usage (main.py script) ---
if __name__ == "__main__":
    # --- Configuration Parameters for Execution ---
    # Base directory where your downloaded daily NetCDF files are located
    DATA_ROOT = Path('data_cds') 
    GEE_DATA_ROOT = Path('data_gee') # Directory for GEE data
    # Directory where you want to save the output GeoTIFFs
    OUTPUT_ROOT = Path('output/hourly_cropped_tiffs/')

    # --- Initialize the Processor ---
    # Note: For this example, we pass DATA_ROOT but it's mainly for the `process_time_series_to_geotiffs` function.
    # The comparison function takes its own data paths.
    processor = ERA5LandProcessor(
        base_data_dir=DATA_ROOT,
        output_base_dir=OUTPUT_ROOT,
        timezone_offset_hours=7 # Vietnam timezone offset
    )

    # --- Example 1: Process a time series to GeoTIFFs ---
    SERIES_START_TIME = '2024-01-01T00:00:00'
    SERIES_END_TIME = '2024-01-01T05:00:00'
    TARGET_BBOX_CLIP = [102.0, 8.0, 110.0, 23.5] # [min_lon, min_lat, max_lon, max_lat] for Vietnam
    BANDS_TO_EXTRACT = ['t2m', 'tp', 'u10', 'v10'] 

    print("\n--- Running Example 1: Time Series GeoTIFF Processing ---")
    saved_geotiffs = processor.process_time_series_to_geotiffs(
        start_timestamp=SERIES_START_TIME,
        end_timestamp=SERIES_END_TIME,
        bbox=TARGET_BBOX_CLIP,
        variables_to_extract=BANDS_TO_EXTRACT
    )
    print(f"Processing finished. Saved {len(saved_geotiffs)} GeoTIFFs.")


    # --- Example 2: Compare CDS vs GEE for a single hour ---
    # Define the specific local hour you want to compare
    COMPARISON_TIMESTAMP_STR = '2024-01-01T08:00:00' # e.g., 8 AM Vietnam time
    comparison_datetime_object = datetime.fromisoformat(COMPARISON_TIMESTAMP_STR)

    print("\n\n--- Running Example 2: CDS vs GEE Comparison ---")
    comparison_results = processor.compare_cds_vs_gee_at_hour(
        comparison_dt=comparison_datetime_object,
        cds_data_dir=DATA_ROOT,
        gee_data_dir=GEE_DATA_ROOT
    )
    if comparison_results:
        print("\nComparison Summary:")
        for var, metrics in comparison_results.items():
            print(f"  Variable: {var}")
            if metrics['compared_pixels'] > 0:
                print(f"    MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, Pixels: {metrics['compared_pixels']}")
            else:
                print("    No comparable data found.")

    print("\nMain execution complete.")