# ERA5 Data Download for Vietnam Grids

This repository contains scripts to download ERA5 climate reanalysis data for Vietnam grid locations using the Copernicus Climate Data Store (CDS) API.

## Files Overview

- `test_cds_api.py` - Simple API test script
- `pipelines.py` - Simplified working version
- `pipelines_final.py` - Fixed version with working variables
- `era5_download_production.py` - Production-ready script with command-line options
- `compare_grib_netcdf.py` - Script to compare GRIB vs NetCDF format efficiency
- `check_variables.py` - Utility to check available variables (not functional)

## Prerequisites

1. **CDS API Setup**: You need to have the CDS API configured with your credentials
2. **Python Dependencies**: 
   ```bash
   pip install cdsapi xarray cfgrib
   ```
3. **GeoJSON File**: `ERA5/Grid_50K_MatchedDates.geojson` containing Vietnam grid polygons

## Quick Start

### 1. Test the API
First, test that your CDS API is working:
```bash
python test_cds_api.py
```

### 2. Run Production Download
Use the production script with default settings (10 grids, 2024-01-01 to 2024-03-01):
```bash
python era5_download_production.py
```

## Production Script Usage

The `era5_download_production.py` script supports various command-line options:

### Basic Usage
```bash
# Default: 10 random grids, 2024-01-01 to 2024-03-01
python era5_download_production.py

# Custom sample size
python era5_download_production.py --sample-size 20

# Custom date range
python era5_download_production.py --start-date 2020-01-01 --end-date 2020-12-31

# Specific grid IDs
python era5_download_production.py --grid-ids 00000000000000000051 00000000000000000052

# Custom data directory
python era5_download_production.py --data-dir /path/to/data

# Adjust delay between requests
python era5_download_production.py --delay 10
```

### Command Line Options

- `--sample-size N`: Number of grids to randomly sample (default: 10)
- `--start-date YYYY-MM-DD`: Start date for data download (default: 2024-01-01)
- `--end-date YYYY-MM-DD`: End date for data download (default: 2024-03-01)
- `--grid-ids ID1 ID2 ...`: Specific grid IDs to download (overrides sample-size)
- `--data-dir PATH`: Data directory (default: data)
- `--delay SECONDS`: Delay between API requests (default: 5)

### Examples

```bash
# Download all grids for 2023
python era5_download_production.py --sample-size 0 --start-date 2023-01-01 --end-date 2023-12-31

# Download specific grids for a short period
python era5_download_production.py --grid-ids 00000000000000000051 --start-date 2024-01-01 --end-date 2024-01-31

# High-volume download with longer delays
python era5_download_production.py --sample-size 50 --delay 10
```

## Data Structure

The downloaded data is organized as follows:
```
data/
└── era5_vietnam/
    └── grib/
        ├── 2024/
        │   ├── 01/
        │   │   ├── 00000000000000000051_2024_01.grib
        │   │   └── 00000000000000000052_2024_01.grib
        │   └── 02/
        │       ├── 00000000000000000051_2024_02.grib
        │       └── 00000000000000000052_2024_02.grib
        └── 2025/
            └── 01/
                ├── 00000000000000000051_2025_01.grib
                └── 00000000000000000052_2025_01.grib
```

## Variables Downloaded

The script downloads the following ERA5 variables (confirmed working):
- `2m_temperature` - 2-meter temperature
- `surface_pressure` - Surface pressure
- `total_precipitation` - Total precipitation

## Data Format

- **Format**: GRIB (Gridded Binary)
- **Temporal Resolution**: Every 6 hours (00:00, 06:00, 12:00, 18:00)
- **Spatial Coverage**: Individual grid polygons from the GeoJSON file
- **Time Period**: All days in the specified month

## Monitoring and Logging

- **Log File**: `era5_download_production.log`
- **Progress Tracking**: Real-time progress updates in the log
- **Error Handling**: Automatic retries with exponential backoff
- **Statistics**: Final summary with success rates

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Usually means invalid variable names
   - Solution: Use only the working variables listed above

2. **API Rate Limiting**: Too many requests too quickly
   - Solution: Increase the `--delay` parameter

3. **Authentication Errors**: CDS API not configured
   - Solution: Set up your CDS API credentials

4. **Memory Issues**: Large downloads
   - Solution: Reduce sample size or date range

### Performance Tips

- Start with small samples to test
- Use longer delays (10+ seconds) for large downloads
- Monitor the log file for progress and errors
- Consider running during off-peak hours

## File Sizes

Typical file sizes per grid per month:
- **Small grids**: ~40-100 KB
- **Large grids**: ~200-500 KB
- **Full Vietnam coverage**: ~50-100 MB per month

## Next Steps

1. **Data Processing**: Use xarray/cfgrib to read the GRIB files
2. **Analysis**: Perform climate analysis on the downloaded data
3. **Visualization**: Create maps and time series plots
4. **Integration**: Combine with other datasets

## Support

For issues with the CDS API or data access, refer to:
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- [CDS API Documentation](https://cds.climate.copernicus.eu/api-how-to)
- [ERA5 Documentation](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) 