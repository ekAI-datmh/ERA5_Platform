import cdsapi
import os
import json
import random
from datetime import datetime, timedelta
import time
import logging
import argparse
import math
import shutil
import tempfile
import subprocess

# Set up logging
logging.basicConfig(
    filename='era5_download_production.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_geojson_grids(geojson_path, sample_size=None, grid_ids=None):
    """
    Read GeoJSON file and extract grids.
    
    Args:
        geojson_path (str): Path to the GeoJSON file.
        sample_size (int): Number of sample grids to extract (if None, use all).
        grid_ids (list): Specific grid IDs to extract (if provided, overrides sample_size).
    
    Returns:
        list: List of grid dictionaries with coordinates and properties.
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    features = data['features']
    
    if grid_ids:
        # Extract specific grids by ID
        selected_features = [f for f in features if f['id'] in grid_ids]
        if len(selected_features) != len(grid_ids):
            missing = set(grid_ids) - set(f['id'] for f in selected_features)
            logging.warning(f"Some grid IDs not found: {missing}")
    elif sample_size:
        # Random sample
        selected_features = random.sample(features, min(sample_size, len(features)))
    else:
        # Use all grids
        selected_features = features
    
    grids = []
    for feature in selected_features:
        coords = feature['geometry']['coordinates'][0]
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        
        grid_info = {
            'id': feature['id'],
            'phien_hieu': feature['properties']['PhienHieu'],
            'min_lon': min(lons),
            'max_lon': max(lons),
            'min_lat': min(lats),
            'max_lat': max(lats),
            'center_lon': sum(lons) / len(lons),
            'center_lat': sum(lats) / len(lats),
            'm_count': feature['properties']['m_count'],
            'm_dates': feature['properties']['m_dates']
        }
        grids.append(grid_info)
    
    return grids

# -------------------- New: CDO-based resampling helpers --------------------

def ensure_cdo() -> str:
    path = shutil.which('cdo')
    return path or ''


def parse_resolution_to_degrees(res_str: str, ref_lat_deg: float) -> float:
    """Parse a resolution string like '0.01deg' or '30m' to degrees.
    For meters, convert using ~111320 m/deg for latitude and adjust longitude by cos(lat).
    Returns the latitude degree step. Longitude degree step can be scaled by cos(lat) later if needed.
    """
    s = res_str.strip().lower()
    if s.endswith('deg'):
        try:
            return float(s.replace('deg', '').strip())
        except Exception:
            raise ValueError(f"Invalid degree resolution: {res_str}")
    if s.endswith('m'):
        try:
            meters = float(s.replace('m', '').strip())
        except Exception:
            raise ValueError(f"Invalid meter resolution: {res_str}")
        # Latitudinal conversion (approx)
        deg_lat = meters / 111320.0
        return deg_lat
    raise ValueError("Resolution must end with 'deg' or 'm', e.g., 0.01deg or 1000m")


def write_cdo_gridfile(grid_path: str, lon_w: float, lon_e: float, lat_s: float, lat_n: float,
                        dlat_deg: float, ref_lat_deg: float) -> None:
    """Write a CDO lon-lat grid description file using the bounding box and resolution.
    Longitude step is adjusted by cos(latitude) to approximately match meter-based requests.
    """
    # Adjust dlon by cos(lat) to approximate meters if input was meters
    cosphi = max(0.1, math.cos(math.radians(ref_lat_deg)))  # avoid extreme near-pole behavior
    dlon_deg = dlat_deg / cosphi

    # Compute number of grid points; ensure at least 1
    nx_float = max(1.0, (lon_e - lon_w) / dlon_deg)
    ny_float = max(1.0, (lat_n - lat_s) / dlat_deg)
    nx = max(1, int(round(nx_float)))
    ny = max(1, int(round(ny_float)))

    # Recompute exact increments to span the bbox
    dlon = (lon_e - lon_w) / nx
    dlat = (lat_n - lat_s) / ny

    # Use lower-left corner as first point; CDO expects centers when using inc
    xfirst = lon_w + dlon / 2.0
    yfirst = lat_s + dlat / 2.0

    with open(grid_path, 'w') as gf:
        gf.write("gridtype = lonlat\n")
        gf.write(f"xsize   = {nx}\n")
        gf.write(f"ysize   = {ny}\n")
        gf.write(f"xfirst  = {xfirst}\n")
        gf.write(f"xinc    = {dlon}\n")
        gf.write(f"yfirst  = {yfirst}\n")
        gf.write(f"yinc    = {dlat}\n")


def cdo_remapbil_to_grid(cdo: str, gridfile: str, infile: str, outfile: str) -> None:
    cmd = [cdo, '-s', '-O', f'remapbil,{gridfile}', infile, outfile]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ---------------------------------------------------------------------------

def generate_date_ranges(start_date, end_date):
    """
    Generate a list of year-month tuples for the given date range.
    
    Args:
        start_date (datetime): Starting date.
        end_date (datetime): Ending date.
    
    Returns:
        list: List of (year, month) tuples.
    """
    date_ranges = []
    current_date = start_date
    while current_date <= end_date:
        date_ranges.append((current_date.year, f"{current_date.month:02d}"))
        next_month = current_date.month % 12 + 1
        next_year = current_date.year + (current_date.month // 12)
        current_date = current_date.replace(year=next_year, month=next_month, day=1)
    return date_ranges

def download_era5_data(dataset, request, target, retries=3):
    """
    Download ERA5 data from CDS with retries.
    
    Args:
        dataset (str): CDS dataset name.
        request (dict): Request parameters.
        target (str): Output file path.
        retries (int): Number of retry attempts.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        client = cdsapi.Client()
        for attempt in range(retries):
            try:
                logging.info(f"Attempt {attempt + 1}: Downloading {target}...")
                start_time = time.time()
                client.retrieve(dataset, request, target)
                end_time = time.time()
                
                if os.path.exists(target):
                    file_size = os.path.getsize(target) / (1024 * 1024)  # MB
                    download_time = end_time - start_time
                    logging.info(f"Success: {target} ({file_size:.2f} MB, {download_time:.2f}s)")
                    return True
                else:
                    logging.warning(f"Failed to download {target}")
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: Error downloading {target}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(10)  # Wait before retry
        return False
    except Exception as e:
        logging.error(f"Error initializing client for {target}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download ERA5 data for Vietnam grids')
    parser.add_argument('--sample-size', type=int, default=10, 
                       help='Number of grids to sample (default: 10)')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date in YYYY-MM-DD format (default: 2024-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-03-01',
                       help='End date in YYYY-MM-DD format (default: 2024-03-01)')
    parser.add_argument('--grid-ids', nargs='+', 
                       help='Specific grid IDs to download')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--delay', type=int, default=5,
                       help='Delay between requests in seconds (default: 5)')
    parser.add_argument('--format', type=str, default='grib', choices=['grib', 'netcdf'],
                       help='Data format to download: grib or netcdf (default: grib)')
    parser.add_argument('--days', type=str, default='',
                       help='Comma-separated list of days (e.g., 01,15,31). If empty, all valid days are used.')
    parser.add_argument('--times', type=str, default='',
                       help='Comma-separated list of times HH:MM (e.g., 00:00,06:00). If empty, 6-hourly steps are used.')
    # New resampling options
    parser.add_argument('--resample', type=str, default='',
                       help="Optional spatial resampling resolution, e.g., '0.01deg' (~1 km) or '1000m'. Applied post-download using CDO bilinear remapping.")
    parser.add_argument('--resample-overwrite', action='store_true',
                       help='Overwrite the original file with resampled output (default: keep both)')
    # New: server-side grid via CDS API
    parser.add_argument('--server-grid', type=str, default='',
                       help="Optional server-side grid in degrees, e.g., '0.25/0.25' or '0.1'. Note: extremely fine values (e.g., 0.001) are not supported by CDS and will fail.")
    args = parser.parse_args()
    
    # Configuration
    geojson_path = 'ERA5/Grid_50K_MatchedDates.geojson'
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    data_dir = args.data_dir

    # CDO path (for optional resampling)
    cdo_path = ensure_cdo() if args.resample else ''
    if args.resample and not cdo_path:
        logging.error("--resample was requested but 'cdo' was not found in PATH. Install with: sudo apt install cdo or conda install -c conda-forge cdo")
        raise SystemExit("CDO not found; cannot perform resampling")

    # Validate/parse server-grid if provided
    server_grid_vals = None
    if args.server_grid:
        raw = args.server_grid.strip()
        if '/' in raw:
            parts = raw.split('/')
        elif ',' in raw:
            parts = raw.split(',')
        else:
            parts = [raw, raw]
        try:
            gx = float(parts[0])
            gy = float(parts[1])
        except Exception:
            raise SystemExit("--server-grid must be numeric like '0.25/0.25' or '0.1'")
        # Basic sanity check: disallow too-fine requests that CDS cannot deliver
        if gx < 0.01 or gy < 0.01:
            raise SystemExit("--server-grid too fine (<0.01°). CDS ERA5 will likely reject it. Use a coarser grid or post-download --resample.")
        server_grid_vals = [gx, gy]

    # Read grids from GeoJSON
    logging.info(f"Reading grids from {geojson_path}...")
    grids = read_geojson_grids(geojson_path, args.sample_size, args.grid_ids)
    logging.info(f"Selected {len(grids)} grids for download")
    
    # Generate date ranges
    date_ranges = generate_date_ranges(start_date, end_date)
    logging.info(f"Date range: {start_date.date()} to {end_date.date()} ({len(date_ranges)} months)")
    
    # Working variables for ERA5 single levels (tested and confirmed working)
    # Note: NetCDF format supports all variables, so we can include more comprehensive data
    working_variables = [
        '2m_temperature', 
        'surface_pressure', 
        'total_precipitation'
    ]
    
    # Set output directory and file extension based on format
    if args.format == 'grib':
        out_dir = os.path.join(data_dir, 'era5_vietnam', 'grib')
        file_ext = 'grib'
    else:
        out_dir = os.path.join(data_dir, 'era5_vietnam', 'netcdf')
        file_ext = 'nc'
    os.makedirs(out_dir, exist_ok=True)

    # Parse optional day/time filters
    days_filter = []
    if args.days:
        days_filter = [d.strip() for d in args.days.split(',') if d.strip()]
    times_filter = []
    if args.times:
        times_filter = [t.strip() for t in args.times.split(',') if t.strip()]
    
    # Statistics
    total_requests = len(grids) * len(date_ranges)
    successful_downloads = 0
    failed_downloads = 0
    
    if args.resample:
        logging.info(f"Resampling requested: {args.resample}. This performs post-download spatial interpolation using CDO bilinear remapping. Note: requesting extremely fine resolutions (e.g., 30 m) can create very large files and be scientifically inappropriate for ERA5 (native ~0.25°).")
    
    logging.info(f"Starting download of {total_requests} files...")
    
    # Download data for each grid
    for grid_idx, grid in enumerate(grids, 1):
        area = [grid['max_lat'], grid['min_lon'], grid['min_lat'], grid['max_lon']]
        grid_id = grid['id']
        
        # Print/log grid coordinates
        coords_msg = (
            f"Downloading grid {grid_idx}/{len(grids)} ID={grid_id} PhienHieu={grid.get('phien_hieu','')} "
            f"bbox(lat,lon): [({grid['min_lat']:.6f}, {grid['min_lon']:.6f}) → ({grid['max_lat']:.6f}, {grid['max_lon']:.6f})] "
            f"area_param(N,W,S,E): [{area[0]:.6f}, {area[1]:.6f}, {area[2]:.6f}, {area[3]:.6f}]"
        )
        print(coords_msg)
        logging.info(coords_msg)
        
        logging.info(f"Processing grid {grid_idx}/{len(grids)}: {grid_id} ({grid['phien_hieu']})")
        
        for date_idx, (year, month) in enumerate(date_ranges, 1):
            # Create request with working variables
            num_days = 31
            try:
                import calendar
                num_days = calendar.monthrange(int(year), int(month))[1]
            except Exception:
                pass
            all_days = [f"{day:02d}" for day in range(1, num_days + 1)]
            req_days = days_filter if days_filter else all_days
            default_times = [f"{hour:02d}:00" for hour in range(0, 24, 6)]
            req_times = times_filter if times_filter else default_times
            request = {
                'product_type': 'reanalysis',
                'variable': working_variables,
                'year': [str(year)],
                'month': [month],
                'day': req_days,
                'time': req_times,
                'area': area,
                'format': args.format
            }
            if server_grid_vals is not None:
                request['grid'] = server_grid_vals
            
            # Target file path
            target_dir = os.path.join(out_dir, str(year), month)
            os.makedirs(target_dir, exist_ok=True)
            target_file = os.path.join(target_dir, f"{grid_id}_{year}_{month}.{file_ext}")
            
            if not os.path.exists(target_file):
                success = download_era5_data('reanalysis-era5-single-levels', request, target_file)
                if success:
                    successful_downloads += 1
                    logging.info(f"Downloaded data: {target_file}")
                else:
                    failed_downloads += 1
                    logging.error(f"Failed to download data: {target_file}")
                    # Skip resampling on failure
                    continue
            else:
                logging.info(f"Skipping existing file: {target_file}")
            
            # Optional: resample spatial resolution using CDO
            if args.resample and os.path.exists(target_file):
                try:
                    # Parse resolution
                    dlat_deg = parse_resolution_to_degrees(args.resample, grid['center_lat'])
                    # Prepare temporary grid description file
                    with tempfile.TemporaryDirectory(prefix='cdo_grid_') as tdir:
                        gridfile = os.path.join(tdir, 'target_grid.txt')
                        write_cdo_gridfile(
                            gridfile,
                            lon_w=grid['min_lon'], lon_e=grid['max_lon'],
                            lat_s=grid['min_lat'], lat_n=grid['max_lat'],
                            dlat_deg=dlat_deg, ref_lat_deg=grid['center_lat']
                        )
                        # Output path
                        if args.resample_overwrite:
                            resampled_path = target_file
                        else:
                            base, ext = os.path.splitext(target_file)
                            resampled_path = f"{base}_resampled{ext}"
                        # Perform remap
                        cdo_remapbil_to_grid(cdo_path, gridfile, target_file, resampled_path)
                        logging.info(f"Resampled to {args.resample}: {resampled_path}")
                        # If overwriting, ensure timestamp update
                        if args.resample_overwrite:
                            os.utime(resampled_path, None)
                except Exception as e:
                    logging.error(f"Resampling failed for {target_file}: {e}")
            
            # Progress update
            current_request = (grid_idx - 1) * len(date_ranges) + date_idx
            progress = (current_request / total_requests) * 100
            logging.info(f"Progress: {current_request}/{total_requests} ({progress:.1f}%)")
            
            # Add delay between requests to avoid overwhelming the API
            time.sleep(args.delay)
    
    # Final summary
    logging.info("=" * 60)
    logging.info("DOWNLOAD SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total grids processed: {len(grids)}")
    logging.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logging.info(f"Total requests: {total_requests}")
    logging.info(f"Successful downloads: {successful_downloads}")
    logging.info(f"Failed downloads: {failed_downloads}")
    logging.info(f"Success rate: {(successful_downloads/total_requests)*100:.1f}%")
    logging.info(f"Data directory: {out_dir}/")

if __name__ == "__main__":
    main() 