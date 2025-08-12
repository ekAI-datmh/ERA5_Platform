# ERA5_Land Data Processing Module

A comprehensive Python module for downloading, preprocessing, and searching ERA5_Land climate data for Vietnam and other regions.

## Overview

This module provides three main functionalities:

1. **Data Download**: Download ERA5_Land data from the Climate Data Store (CDS) for Vietnam
2. **Data Preprocessing**: Resample, mask, and convert downloaded data to target formats
3. **Data Search**: Extract data for specific regions and time periods from the downloaded database

## Features

- **Efficient Data Storage**: Organized storage schema (`data/era5_vietnam/grib|netcdf/year/month/`)
- **Parallel Processing**: Multi-threaded downloads and processing for improved performance
- **Flexible Formats**: Support for GRIB, NetCDF, and GeoTIFF formats
- **Grid Masking**: Mask areas outside defined grids using GeoJSON data
- **High-Resolution Output**: Resample data to target resolutions (e.g., 30m)
- **Comprehensive Logging**: Detailed logging and progress tracking
- **Error Handling**: Robust error handling and retry mechanisms

## Installation

### Prerequisites

1. **Python 3.8+**
2. **CDS API Key**: Register at [Climate Data Store](https://cds.climate.copernicus.eu/) and get your API key
3. **CDO (Climate Data Operators)**: For advanced data processing (optional but recommended)

### Setup

1. **Clone or download the module**:
   ```bash
   # If you have the module files, place them in your project directory
   ```

2. **Install required Python packages**:
   ```bash
   pip install cdsapi xarray numpy pandas geopandas rasterio pygrib shapely tqdm
   ```

3. **Set up CDS API key**:
   ```bash
   # Option 1: Environment variable
   export CDS_API_KEY="your-api-key-here"
   
   # Option 2: Create .cdsapirc file in your home directory
   echo "key: your-api-key-here" > ~/.cdsapirc
   ```

4. **Install CDO (optional but recommended)**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cdo
   
   # macOS
   brew install cdo
   
   # Conda
   conda install -c conda-forge cdo
   ```

## Quick Start

### 1. Download ERA5_Land Data

```python
from era5_land_module import ERA5Downloader

# Initialize downloader
downloader = ERA5Downloader(
    dataset_type='hourly',
    format_type='grib',
    variables=['2m_temperature', 'total_precipitation']
)

# Download data for January 2024
stats = downloader.download_data(
    start_date='2024-01-01',
    end_date='2024-01-31'
)

print(f"Downloaded {stats['successful_downloads']} files")
```

### 2. Preprocess Downloaded Data

```python
from era5_land_module import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    target_resolution='30m',
    interpolation_method='bilinear',
    mask_outside_grids=True,
    output_format='tif'
)

# Preprocess downloaded data
stats = preprocessor.preprocess_directory(
    input_dir='data/era5_vietnam/grib',
    output_dir='data/era5_vietnam/processed'
)

print(f"Processed {stats['processed_files']} files")
```

### 3. Search and Extract Data

```python
from era5_land_module import DataSearcher

# Initialize searcher
searcher = DataSearcher(
    data_dir='data/era5_vietnam',
    output_resolution='30m'
)

# Search for data in specific region
bbox = ((105.0, 16.0), (110.0, 10.0))  # (top_left, bottom_right)
stats = searcher.search_data(
    bbox=bbox,
    start_date='2024-01-01',
    end_date='2024-01-31',
    variables=['2m_temperature']
)

print(f"Extracted {stats['files_extracted']} files")
```

## Command Line Usage

The module includes a command-line interface for easy usage:

### Download Data

```bash
# Download hourly data for January 2024
python main.py --task download \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --format grib \
    --variables 2m_temperature total_precipitation

# Download daily data
python main.py --task download \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --dataset-type daily \
    --format netcdf
```

### Preprocess Data

```bash
# Resample to 30m resolution and apply grid mask
python main.py --task preprocess \
    --input-dir data/era5_vietnam/grib \
    --resolution 30m \
    --interpolation bilinear \
    --mask-grids \
    --output-format tif

# Process specific variables
python main.py --task preprocess \
    --input-dir data/era5_vietnam/grib \
    --variables 2m_temperature total_precipitation \
    --resolution 0.1deg
```

### Search Data

```bash
# Search for data in specific region
python main.py --task search \
    --bbox 105,16,110,10 \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --variables 2m_temperature

# Search with custom parameters
python main.py --task search \
    --bbox 105,16,110,10 \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --max-bbox-size 3.0 \
    --time-buffer 2 \
    --output-resolution 30m
```

## Data Storage Schema

The module uses an organized storage schema:

```
data/
└── era5_vietnam/
    ├── grib/                    # GRIB format files
    │   ├── 2024/
    │   │   ├── 01/             # January
    │   │   │   ├── 2m_temperature_2024_01_01_00.grib
    │   │   │   ├── 2m_temperature_2024_01_01_01.grib
    │   │   │   └── ...
    │   │   └── 02/             # February
    │   └── ...
    ├── netcdf/                  # NetCDF format files
    │   └── ...
    ├── processed/               # Preprocessed files
    │   ├── 2m_temperature_2024_01_01_00_processed.tif
    │   └── ...
    └── output/                  # Search results
        ├── 2m_temperature_105.000_10.000_110.000_16.000_20240101_0000.tif
        └── ...
```

## Configuration

The module uses a centralized configuration system. Key configuration options:

### ERA5_Land Variables

**Hourly Variables**:
- `2m_temperature` - 2-meter temperature
- `2m_relative_humidity` - 2-meter relative humidity
- `surface_pressure` - Surface pressure
- `total_precipitation` - Total precipitation
- `10m_u_component_of_wind` - 10-meter U wind component
- `10m_v_component_of_wind` - 10-meter V wind component
- And many more...

**Daily Variables**:
- `2m_temperature_max` - Maximum 2-meter temperature
- `2m_temperature_min` - Minimum 2-meter temperature
- `total_precipitation` - Total precipitation
- And more...

### Vietnam Bounding Box

Default Vietnam bounding box:
- North: 23.5°N
- South: 8.5°N
- West: 102.0°E
- East: 110.0°E

### Processing Options

- **Target Resolution**: 30m, 0.1deg, etc.
- **Interpolation Methods**: bilinear, nearest, cubic
- **Fill Methods**: interpolate, fill, none
- **Output Formats**: tif, grib, netcdf
- **Compression**: lzw, deflate, none

## Advanced Usage

### Custom Bounding Box

```python
# Define custom bounding box for specific region
custom_bbox = {
    'north': 22.0,
    'south': 10.0,
    'west': 103.0,
    'east': 109.0
}

downloader = ERA5Downloader(bbox=custom_bbox)
```

### Parallel Processing

```python
# Increase parallel workers for faster processing
downloader = ERA5Downloader(max_workers=8)
preprocessor = DataPreprocessor(max_workers=8)
searcher = DataSearcher(max_workers=8)
```

### Progress Tracking

```python
from tqdm import tqdm

# Custom progress callback
def progress_callback(completed):
    print(f"Processed {completed} items")

# Use in processing
stats = downloader.download_data(
    start_date='2024-01-01',
    end_date='2024-01-31',
    progress_callback=progress_callback
)
```

### Error Handling

```python
try:
    stats = downloader.download_data(
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
except Exception as e:
    print(f"Download failed: {e}")
    # Check failed downloads
    print(f"Failed downloads: {stats['failed_downloads']}")
```

## Troubleshooting

### Common Issues

1. **CDS API Key Not Found**:
   ```bash
   # Set environment variable
   export CDS_API_KEY="your-key-here"
   ```

2. **CDO Not Found**:
   ```bash
   # Install CDO
   sudo apt-get install cdo  # Ubuntu/Debian
   brew install cdo          # macOS
   ```

3. **Memory Issues**:
   - Reduce `max_workers` parameter
   - Process smaller date ranges
   - Use smaller bounding boxes

4. **Disk Space**:
   - Monitor disk usage during downloads
   - Use compression options
   - Clean up temporary files

### Performance Optimization

1. **Download Optimization**:
   - Use appropriate `delay_between_requests`
   - Increase `max_workers` (but not too much)
   - Download smaller date ranges

2. **Processing Optimization**:
   - Use CDO when available
   - Process in chunks
   - Use appropriate resolution settings

3. **Search Optimization**:
   - Use specific variables instead of all
   - Limit bounding box size
   - Use time buffers appropriately

## API Reference

### ERA5Downloader

Main class for downloading ERA5_Land data.

**Methods**:
- `download_data(start_date, end_date, output_dir=None)`: Download data for date range
- `download_variable(variable, start_date, end_date, output_dir=None)`: Download single variable
- `download_month(year, month, variables=None, output_dir=None)`: Download month data
- `get_download_status(output_dir)`: Get download status
- `validate_downloads(output_dir)`: Validate downloaded files

### DataPreprocessor

Main class for preprocessing ERA5_Land data.

**Methods**:
- `preprocess_directory(input_dir, output_dir=None, variables=None)`: Preprocess directory
- `preprocess_file(input_file, output_file=None)`: Preprocess single file

### DataSearcher

Main class for searching and extracting ERA5_Land data.

**Methods**:
- `search_data(bbox, start_date, end_date, variables=None, output_dir=None)`: Search and extract data
- `search_variable(variable, bbox, start_date, end_date, output_dir=None)`: Search single variable
- `get_search_status(output_dir)`: Get search status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- European Centre for Medium-Range Weather Forecasts (ECMWF) for ERA5_Land data
- Climate Data Store (CDS) for data access
- CDO developers for climate data processing tools

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Create an issue with detailed information
4. Include system information and error logs 