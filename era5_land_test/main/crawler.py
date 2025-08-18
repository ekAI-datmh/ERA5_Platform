import os
import cdsapi
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import zipfile
import shutil
import calendar
from pathlib import Path
from datetime import datetime, timedelta
import time
import xarray as xr # Added for merging NetCDF files

# --- GEE Specific Imports ---
import ee # Google Earth Engine API
import rasterio # For handling GeoTIFFs locally
import requests # For downloading files from URL

# Define GEE dataset name for ERA5-Land Hourly
# Source: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY
ERA5_LAND_GEE_DATASET = 'ECMWF/ERA5_LAND/HOURLY'

# Mapping from CDS API variable names (long form) to GEE band names
# GEE generally uses the long form for band names in ERA5-Land, but sometimes with minor differences
# Note: total_precipitation is called total_precipitation_hourly in GEE for this dataset
GEE_VARIABLE_MAP = {
    "2m_dewpoint_temperature": "dewpoint_temperature_2m",
    "2m_temperature": "temperature_2m",
    "skin_temperature": "skin_temperature",
    "surface_latent_heat_flux": "surface_latent_heat_flux",
    "surface_net_solar_radiation": "surface_net_solar_radiation",
    "surface_sensible_heat_flux": "surface_sensible_heat_flux",
    "surface_solar_radiation_downwards": "surface_solar_radiation_downwards",
    "potential_evaporation": "potential_evaporation",
    "total_evaporation": "total_evaporation",
    "10m_u_component_of_wind": "u_component_of_wind_10m",
    "10m_v_component_of_wind": "v_component_of_wind_10m",
    "surface_pressure": "surface_pressure",
    "total_precipitation": "total_precipitation_hourly" # GEE specific name for this variable
}

def generate_local_dates(year: int):
    """
    Generates datetime objects for each day in a given year,
    representing the start of the local day (00:00).
    """
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += timedelta(days=1)


# --- CDS API Crawler Class ---
class CDSAPICrawler:
    """
    Crawler for ERA5-Land data from Copernicus Climate Data Store (CDSAPI).
    """
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, env_file: Optional[str] = None):
        load_dotenv(env_file)
        self.url = url or os.getenv('CDSAPI_URL', 'https://cds.climate.copernicus.eu/api')
        self.key = key or os.getenv('CDSAPI_KEY')
        if not self.key:
            raise ValueError("CDS API key is required. Set CDSAPI_KEY environment variable or pass key parameter.")
    
    def get_client(self) -> cdsapi.Client:
        return cdsapi.Client(url=self.url, key=self.key)

    def _unzip_and_move(self, raw_path: Path, final_path: Path):
        """
        Helper function to handle potential ZIP extraction for a downloaded file.
        Moves the final NetCDF to `final_path` and cleans up the raw file.
        Returns True if successful, False otherwise.
        """
        if not raw_path.exists():
            print(f"  Raw file not found: {raw_path}")
            return False
        
        try:
            if zipfile.is_zipfile(raw_path):
                print(f"  Extracting ZIP file: {raw_path}...")
                with zipfile.ZipFile(raw_path, 'r') as zip_ref:
                    nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                    if nc_files:
                        zip_ref.extract(nc_files[0], path=raw_path.parent)
                        extracted_temp_path = raw_path.parent / nc_files[0]
                        shutil.move(extracted_temp_path, final_path)
                        print(f"  Extracted to {final_path}")
                        return True
                    else:
                        print(f"  No NetCDF file found inside {raw_path}")
                        return False
            else:
                shutil.move(raw_path, final_path)
                print(f"  Downloaded directly as NetCDF: {final_path}")
                return True
        except Exception as e:
            print(f"  Error during unzip/move for {raw_path}: {e}")
            return False
        finally:
            if raw_path.exists():
                os.remove(raw_path)

    def download_era5_land_series(
        self,
        year: int,
        base_output_dir: Path, # Use for local save
        variables_cds: list, # Use variables_cds for clarity (CDS specific names)
        area: list,
        hourly_interval: int = 1, # Default to all 24 hours per day
        time_delay_seconds: int = 2,
        timezone_offset_hours: int = 0, # Offset from UTC (e.g., 7 for Vietnam)
        months_to_crawl: Optional[list[int]] = None # New: List of months (1-12) to crawl, defaults to all if None
    ):
        client = self.get_client()
        
        print(f"\n--- Starting CDS API Download for year {year} ---")
        print(f"  Output directory: {base_output_dir}")
        print(f"  Variables: {variables_cds}")
        print(f"  Area: {area}")
        print(f"  Timezone Offset: UTC{'+' if timezone_offset_hours >= 0 else ''}{timezone_offset_hours} hours")
        print(f"  Note: CDS data will cover 00:00-23:59 local time, spanning UTC days as needed.")
        print(f"  Months to crawl: {months_to_crawl if months_to_crawl else 'All months'}")
        
        # Ensure base output directory exists for local files
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate through specified months, or all 12 if not provided
        for local_date in generate_local_dates(year):
            if months_to_crawl and local_date.month not in months_to_crawl:
                continue # Skip if current month is not in the list

            month_output_dir = base_output_dir / str(local_date.year) / f"{local_date.month:02d}"
            month_output_dir.mkdir(parents=True, exist_ok=True)

            output_filename_extracted = month_output_dir / f'era5_vietnam_{local_date.strftime("%Y_%m_%d")}.nc'

            print(f"Processing for local date {local_date.strftime('%Y-%m-%d')}... ")

            if output_filename_extracted.exists():
                print(f"File already exists: {output_filename_extracted}. Skipping download.")
                continue

            # Calculate UTC start and end datetimes for the 00:00-23:59 local day
            utc_start_request_dt = local_date - timedelta(hours=timezone_offset_hours)
            utc_end_request_dt = (local_date + timedelta(days=1)) - timedelta(hours=timezone_offset_hours) - timedelta(seconds=1)

            temp_files_to_clean = []
            try:
                # Case 1: The entire 24h local period is within one UTC day (e.g., timezone_offset_hours = 0)
                if utc_start_request_dt.date() == utc_end_request_dt.date():
                    print(f"  Requesting single UTC day: {utc_start_request_dt.strftime('%Y-%m-%d')}")
                    utc_hours_for_request = [f'{h:02d}:00' for h in range(24)]
                    raw_path = month_output_dir / "temp_single_raw.nc"
                    temp_files_to_clean.append(raw_path)
                    
                    client.retrieve(
                        'reanalysis-era5-land', {
                            'variable': variables_cds, 'year': str(utc_start_request_dt.year),
                            'month': f'{utc_start_request_dt.month:02d}', 'day': f'{utc_start_request_dt.day:02d}',
                            'time': utc_hours_for_request, 'area': area, 'grid': [0.1, 0.1],
                            'format': 'netcdf', 'product_type': 'reanalysis'
                        }, str(raw_path)
                    )
                    self._unzip_and_move(raw_path, output_filename_extracted)

                # Case 2: The 24h local period spans two UTC days (the fix for the 48h bug)
                else:
                    print(f"  Request spans two UTC days: {utc_start_request_dt.strftime('%Y-%m-%d')} and {utc_end_request_dt.strftime('%Y-%m-%d')}")
                    # Part 1: Get hours for the first UTC day
                    hours_day1 = [f'{h:02d}:00' for h in range(utc_start_request_dt.hour, 24)]
                    temp_raw_1 = month_output_dir / "temp_part1_raw.nc"
                    temp_files_to_clean.append(temp_raw_1)
                    
                    client.retrieve(
                        'reanalysis-era5-land', {
                            'variable': variables_cds, 'year': str(utc_start_request_dt.year),
                            'month': f'{utc_start_request_dt.month:02d}', 'day': f'{utc_start_request_dt.day:02d}',
                            'time': hours_day1, 'area': area, 'grid': [0.1, 0.1],
                            'format': 'netcdf', 'product_type': 'reanalysis'
                        }, str(temp_raw_1)
                    )

                    # Part 2: Get hours for the second UTC day
                    hours_day2 = [f'{h:02d}:00' for h in range(0, utc_end_request_dt.hour + 1)]
                    temp_raw_2 = month_output_dir / "temp_part2_raw.nc"
                    temp_files_to_clean.append(temp_raw_2)

                    client.retrieve(
                        'reanalysis-era5-land', {
                            'variable': variables_cds, 'year': str(utc_end_request_dt.year),
                            'month': f'{utc_end_request_dt.month:02d}', 'day': f'{utc_end_request_dt.day:02d}',
                            'time': hours_day2, 'area': area, 'grid': [0.1, 0.1],
                            'format': 'netcdf', 'product_type': 'reanalysis'
                        }, str(temp_raw_2)
                    )

                    # Unzip and prepare for merge
                    temp_nc_1 = month_output_dir / "temp_part1.nc"
                    temp_nc_2 = month_output_dir / "temp_part2.nc"
                    temp_files_to_clean.extend([temp_nc_1, temp_nc_2])

                    self._unzip_and_move(temp_raw_1, temp_nc_1)
                    self._unzip_and_move(temp_raw_2, temp_nc_2)

                    # Merge the two NetCDF files
                    if temp_nc_1.exists() and temp_nc_2.exists():
                        print(f"  Merging temporary files into {output_filename_extracted.name}...")
                        with xr.open_mfdataset([str(temp_nc_1), str(temp_nc_2)]) as ds:
                            ds.to_netcdf(str(output_filename_extracted))
                        print(f"  Successfully merged and saved.")
                    else:
                        raise IOError("One or both temporary NetCDF parts were not created, cannot merge.")

            except Exception as e:
                print(f"Error processing {local_date.strftime('%Y-%m-%d')}: {e}")
            finally:
                # Clean up all temporary files
                for f in temp_files_to_clean:
                    if f.exists():
                        try:
                            os.remove(f)
                        except OSError as e:
                            print(f"  Error removing temp file {f}: {e}")
                time.sleep(time_delay_seconds)

        print(f"--- Finished CDS API Download for year {year} ---")


# --- Google Earth Engine Crawler Class ---
class GEECrawler:
    """
    Crawler for ERA5-Land data from Google Earth Engine (GEE).
    This class directly downloads GeoTIFFs locally by initiating GEE export URLs.
    """
    def __init__(self):
        # Authenticate and initialize GEE
        try:
            ee.Initialize() # Using default project from ee.Authenticate()
        except Exception:
            print("GEE not initialized. Please run `ee.Authenticate()` and `ee.Initialize()` manually in a Python console or new Jupyter cell.")
            print("Or, if using a service account, ensure GOOGLE_APPLICATION_CREDENTIALS is set and ee.Initialize() is called with project.")
            # If not initialized, subsequent GEE calls will fail. Consider raising an error here if critical.

    def _combine_tifs_to_multiband(self, temp_dir: Path, tif_files: list, bands_order: list, output_path: Path):
        """
        Combines individual single-band GeoTIFFs into a single multi-band GeoTIFF.
        """
        if not tif_files:
            print(f"No GeoTIFF files to combine in {temp_dir}")
            return False

        # --- ROBUST BAND ORDERING FIX ---
        # 1. Find all .tif files and extract their GEE band names from filenames.
        #    e.g., from '.../export.temperature_2m.tif' -> 'temperature_2m'
        found_bands_map = {} # dict of clean_band_name -> full_path
        for tif_path_str in tif_files:
            path_obj = Path(tif_path_str)
            # GEE filenames are like `export.temperature_2m.tif`
            band_name_from_file = path_obj.stem.split('.')[-1]
            found_bands_map[band_name_from_file] = temp_dir / path_obj

        # 2. Create the ordered lists based on the original `bands_order` request,
        #    ensuring data and descriptions are perfectly synchronized.
        ordered_tifs = []
        ordered_gee_band_names = []
        for requested_band in bands_order:
            if requested_band in found_bands_map:
                ordered_tifs.append(found_bands_map[requested_band])
                ordered_gee_band_names.append(requested_band) # Use the original name (e.g. with _hourly)

        if len(ordered_tifs) != len(found_bands_map):
             print(f"  Warning: Mismatch between requested bands ({len(bands_order)}) and found tif files ({len(found_bands_map)}). "
                   f"Proceeding with {len(ordered_tifs)} matched bands.")
        # --- END OF FIX ---

        if not ordered_tifs:
            print(f"No matching GeoTIFF files found for requested bands in {temp_dir}.")
            return False
            
        # Create a reverse map to get CDS-style names from GEE band names
        reverse_gee_map = {v: k for k, v in GEE_VARIABLE_MAP.items()}

        try:
            # Read the profile from the first image
            with rasterio.open(ordered_tifs[0]) as src0:
                profile = src0.profile
                band_arrays = [src0.read(1)] # Read the first band

            # Read subsequent bands
            for tif_path in ordered_tifs[1:]:
                with rasterio.open(tif_path) as src:
                    band_arrays.append(src.read(1))
            
            # Update profile for multi-band output
            profile.update(
                count=len(band_arrays), # Set number of bands
                compress='lzw' # LZW compression is good for GeoTIFFs
            )

            # Write to the final output path
            with rasterio.open(output_path, 'w', **profile) as dst:
                for idx, arr in enumerate(band_arrays, start=1):
                    dst.write(arr, idx)
                    # Set band description using the original CDS variable name
                    if idx - 1 < len(ordered_gee_band_names):
                        gee_band_name = ordered_gee_band_names[idx-1]
                        # Get the original CDS variable name, or fallback to GEE name if not found
                        cds_variable_name = reverse_gee_map.get(gee_band_name, gee_band_name)
                        dst.set_band_description(idx, cds_variable_name)
            return True
        except Exception as e:
            print(f"Error combining GeoTIFFs for {output_path.name}: {e}")
            return False


    def download_era5_land_series(
        self,
        year: int,
        base_output_dir: Path, # Local output directory for GEE downloads
        variables_cds: list, # CDS-style long variable names
        area: list, # [north, west, south, east] (will be converted to GEE format)
        hourly_interval: int = 1, # GEE provides hourly data, so typically 1.
        time_delay_seconds: int = 0.5,
        timezone_offset_hours: int = 0, # Offset from UTC (e.g., 7 for Vietnam)
        months_to_crawl: Optional[list[int]] = None # New: List of months (1-12) to crawl, defaults to all if None
    ):

        print(f"\n--- Starting GEE Direct Download for year {year} ---")
        print(f"  Output directory: {base_output_dir}")
        print(f"  Variables (CDS names): {variables_cds}")
        print(f"  Area: {area}")
        print(f"  Timezone Offset: UTC{'+' if timezone_offset_hours >= 0 else ''}{timezone_offset_hours} hours")
        print(f"  Note: GEE data will cover 00:00-23:59 local time.")
        print(f"  Months to crawl: {months_to_crawl if months_to_crawl else 'All months'}")

        # Ensure local base output directory exists
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Convert CDS area [N, W, S, E] to GEE geometry [min_lon, min_lat, max_lon, max_lat]
        # GEE expects [west, south, east, north] for ee.Geometry.Rectangle
        bbox_gee = [area[1], area[2], area[3], area[0]] # [west, south, east, north]
        region_geometry = ee.Geometry.Rectangle(bbox_gee)

        # Map CDS variable names to GEE band names
        gee_bands = [GEE_VARIABLE_MAP.get(v, v) for v in variables_cds]
        print(f"  GEE Bands to request: {gee_bands}")

        # Get the ERA5-Land Image Collection
        era5_land = ee.ImageCollection(ERA5_LAND_GEE_DATASET)

        for local_date in generate_local_dates(year):
            if months_to_crawl and local_date.month not in months_to_crawl:
                continue # Skip if current month is not in the list

            month_output_dir = base_output_dir / str(local_date.year) / f"{local_date.month:02d}"
            month_output_dir.mkdir(parents=True, exist_ok=True)

            # Define the UTC date range for the current local day
            utc_start_request_dt = local_date - timedelta(hours=timezone_offset_hours)
            # GEE's filterDate has an EXCLUSIVE end date. So, to get all data for a 24-hour local day,
            # the end of the filter range should be exactly 24 hours after the start.
            utc_filter_end_dt = utc_start_request_dt + timedelta(days=1)

            print(f"Processing GEE data for local date {local_date.strftime('%Y-%m-%d')}...")

            daily_images_collection = era5_land.filterDate(utc_start_request_dt.strftime('%Y-%m-%dT%H:%M:%S'), 
                                                         utc_filter_end_dt.strftime('%Y-%m-%dT%H:%M:%S')) \
                                            .filterBounds(region_geometry) \
                                            .select(gee_bands)
            
            # Get list of images for the day
            image_list = daily_images_collection.toList(daily_images_collection.size())
            num_images = image_list.size().getInfo()

            if num_images == 0:
                print(f"No GEE hourly data found for {local_date.strftime('%Y-%m-%d')}. Skipping.")
                continue
            
            print(f"Found {num_images} hourly images for {local_date.strftime('%Y-%m-%d')}. Downloading...")

            temp_download_dir = month_output_dir / 'temp_gee_downloads'
            temp_download_dir.mkdir(exist_ok=True)

            for i in range(num_images):
                image = ee.Image(image_list.get(i))
                
                # Get the timestamp for naming (this naturally converts to local time based on system settings)
                image_time_millis = image.get('system:time_start').getInfo()
                image_datetime = datetime.fromtimestamp(image_time_millis / 1000)
                
                # Construct filename for each hourly GeoTIFF based on local time
                output_filename_tif = month_output_dir / f'era5_vietnam_GEE_{image_datetime.strftime("%Y%m%dT%H%M")}.tif'
                
                if output_filename_tif.exists():
                    print(f"File already exists: {output_filename_tif.name}. Skipping.")
                    continue

                try:
                    # GEE exports as single-band TIFFs within a ZIP for `getDownloadURL`
                    # The scale (resolution) should be in meters per pixel. ERA5 Land is roughly 11km. 
                    # Using a high scale like 11132 (approx 0.1 degree at equator) ensures native resolution.
                    download_url = image.getDownloadURL({
                        'scale': 11132, 
                        'region': region_geometry.getInfo(), 
                        'fileFormat': 'GeoTIFF', 
                        'formatDict': {'compression': 'LZW', 'tiled': True, 'file_per_band': True}
                    })

                    # Download the zipped file
                    response = requests.get(download_url, stream=True, timeout=600) # Increased timeout
                    response.raise_for_status() # Raise an exception for bad status codes

                    temp_zip_path = temp_download_dir / f'temp_download_{image_datetime.strftime("%H%M")}.zip'
                    with open(temp_zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Extract the individual band GeoTIFFs
                    extracted_tifs = []
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            if member.endswith('.tif'):
                                zip_ref.extract(member, path=temp_download_dir)
                                extracted_tifs.append(member)
                    os.remove(temp_zip_path) # Clean up the zip file

                    # Combine individual GeoTIFFs into a single multi-band GeoTIFF
                    self._combine_tifs_to_multiband(
                        temp_download_dir,
                        extracted_tifs,
                        gee_bands,
                        output_filename_tif
                    )
                    
                except ee.EEException as e:
                    print(f"GEE Error for {image_datetime.strftime('%Y-%m-%d %H:%M')}: {e}")
                except requests.exceptions.RequestException as e:
                    print(f"HTTP Request Error for {image_datetime.strftime('%Y-%m-%d %H:%M')}: {e}")
                except Exception as e:
                    print(f"Unexpected Error for {image_datetime.strftime('%Y-%m-%d %H:%M')}: {e}")
                finally:
                    # Clean up temporary extracted files for this hour
                    if temp_download_dir.exists():
                        shutil.rmtree(temp_download_dir)
                    temp_download_dir.mkdir(exist_ok=True) # Recreate for next iteration
                    time.sleep(time_delay_seconds)
            
            # Clean up the temporary download directory for the day
            if temp_download_dir.exists():
                shutil.rmtree(temp_download_dir)

        print(f"--- Finished GEE Direct Download for year {year} ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration Parameters for Execution ---
    download_year = 2024
    
    # Define output root directories for CDS and GEE data
    cds_output_root_dir = Path('data_cds') # Local folder for CDS data
    gee_output_root_dir = Path('data_gee') # Local folder for GEE data
    
    # Common region for both crawlers: Vietnam bounding box [north, west, south, east]
    vietnam_bbox_area = [23.5, 102.0, 8.0, 110.0] 
    
    # All variables requested previously (use their long form for CDS API)
    era5_land_variables_long_form = [
        "2m_dewpoint_temperature", "2m_temperature", "skin_temperature",
        "surface_latent_heat_flux", "surface_net_solar_radiation", "surface_sensible_heat_flux",
        "surface_solar_radiation_downwards", "potential_evaporation", "total_evaporation",
        "10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure",
        "total_precipitation"
    ]

    # Example: Crawl only January and February
    months_to_download = [1] # Add this line

    # --- Initialize CDS API Crawler ---
    print("Initializing CDS API Crawler...")
    cds_crawler = CDSAPICrawler(env_file='.env')

    # --- Run CDS API Download (uncomment to enable) ---
    print("\n--- Running CDS API Data Download ---")
    cds_crawler.download_era5_land_series(
        year=download_year,
        base_output_dir=cds_output_root_dir,
        variables_cds=era5_land_variables_long_form,
        area=vietnam_bbox_area,
        hourly_interval=1, # Download all 24 hours per day
        time_delay_seconds=2,
        timezone_offset_hours=7, # Vietnam timezone offset
        months_to_crawl=months_to_download # Add this line
    )
    print("CDS API download process initiated/completed.")

    # # --- Initialize GEE Crawler ---
    # print("\nInitializing GEE Crawler...")
    # gee_crawler = GEECrawler()

    # # --- Run GEE Download (initiates export tasks) ---
    # print("\n--- Starting GEE Direct Data Download ---")
    # gee_crawler.download_era5_land_series(
    #     year=download_year,
    #     base_output_dir=gee_output_root_dir,
    #     variables_cds=era5_land_variables_long_form,
    #     area=vietnam_bbox_area,
    #     hourly_interval=1, # Ensure GEE downloads all hourly data
    #     time_delay_seconds=0.5,
    #     timezone_offset_hours=7, # Vietnam timezone offset
    #     months_to_crawl=months_to_download # Add this line
    # )
    print("GEE direct download process initiated/completed.")

    print("\nMain execution complete.")
