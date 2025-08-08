import cdsapi
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_cds_api():
    """
    Test the CDS API with a minimal request to ensure it's working.
    """
    try:
        client = cdsapi.Client()
        logging.info("CDS API client initialized successfully")
        
        # Test with a minimal request for a small area and short time period
        request = {
            'product_type': 'reanalysis',
            'variable': ['2m_temperature'],
            'year': ['2024'],
            'month': ['01'],
            'day': ['01'],
            'time': ['12:00'],
            'area': [23.5, 102, 8.5, 109],  # Vietnam area
            'format': 'grib'
        }
        
        target_file = 'test_era5_download.grib'
        
        logging.info("Starting test download...")
        client.retrieve('reanalysis-era5-single-levels', request, target_file)
        
        if os.path.exists(target_file):
            file_size = os.path.getsize(target_file) / (1024 * 1024)  # MB
            logging.info(f"Test successful! File downloaded: {target_file} ({file_size:.2f} MB)")
            return True
        else:
            logging.error("Test failed: File was not created")
            return False
            
    except Exception as e:
        logging.error(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_cds_api()
    if success:
        print("CDS API test PASSED - the API is working correctly")
    else:
        print("CDS API test FAILED - check your configuration") 