# ERA5-Land Data Download with CDS API

This project demonstrates how to download ERA5-Land data using the CDS API with multiple configuration methods, including `.env` file support.

## Features

- ✅ Multiple credential configuration methods
- ✅ `.env` file support
- ✅ Environment variable support
- ✅ Direct configuration
- ✅ Custom config file support
- ✅ Comprehensive error handling
- ✅ Type hints and logging

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your CDS API credentials (see Configuration section below)

## Configuration Methods

### Method 1: .env File (Recommended)

Create a `.env` file in your project directory:

```bash
# .env
CDSAPI_URL=https://cds.climate.copernicus.eu/api
CDSAPI_KEY=your-api-key-here
```

The script will automatically load this file when you create a `CDSAPIConfig()` instance.

### Method 2: Environment Variables

Set environment variables in your shell:

```bash
export CDSAPI_URL='https://cds.climate.copernicus.eu/api'
export CDSAPI_KEY='your-api-key-here'
```

### Method 3: Direct Configuration

Pass credentials directly in code:

```python
config = CDSAPIConfig(
    url='https://cds.climate.copernicus.eu/api',
    key='your-api-key-here'
)
```

### Method 4: Custom .env File

Specify a custom .env file path:

```python
config = CDSAPIConfig(env_file='./custom.env')
```

## Usage Examples

### Basic Usage with .env File

```python
from era5_data_download import download_era5_land_data

# Automatically uses .env file if present
success = download_era5_land_data(
    variable='2m_temperature',
    year=2023,
    month=1,
    day=1,
    time=['00:00', '06:00', '12:00', '18:00'],
    area=[60, -10, 50, 2],  # Europe
    output_file='era5_land_temp_20230101.nc'
)
```

### Advanced Configuration

```python
from era5_data_download import CDSAPIConfig, download_era5_land_data

# Load from specific .env file
config = CDSAPIConfig(env_file='./production.env')

# Use the configuration
success = download_era5_land_data(
    variable='2m_temperature',
    year=2023,
    month=1,
    day=1,
    time=['00:00', '06:00', '12:00', '18:00'],
    area=[60, -10, 50, 2],
    output_file='era5_land_temp_20230101.nc',
    config=config
)
```

## File Structure

```
era5_land_test/
├── era5_data_download.py    # Main download script
├── demo_env_usage.py        # Demo script for .env usage
├── requirements.txt         # Python dependencies
├── env_example.txt          # Example .env file format
├── config_example.txt       # Alternative config format
└── README.md               # This file
```

## Security Best Practices

1. **Never commit .env files to version control**
2. **Use different credentials for development and production**
3. **Set appropriate file permissions on .env files**
4. **Consider using secret management services in production**

## Getting CDS API Credentials

1. Register at [CDS](https://cds.climate.copernicus.eu/)
2. Go to your profile page
3. Copy your API key
4. Add it to your `.env` file or environment variables

## Running the Examples

```bash
# Run the main example
python era5_data_download.py

# Run the .env demo
python demo_env_usage.py
```

## Troubleshooting

### Common Issues

1. **"CDS API key is required"**: Make sure your `.env` file exists and contains `CDSAPI_KEY`
2. **"Invalid credentials"**: Verify your API key is correct
3. **"Network error"**: Check your internet connection and firewall settings

### Debug Mode

Enable debug logging by modifying the logging level in `era5_data_download.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is open source and available under the MIT License. 