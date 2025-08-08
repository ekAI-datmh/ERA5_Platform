import cdsapi
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_era5_data(dataset, request, target_file):
    """
    Downloads ERA5 data using cdsapi with efficiency in mind.

    This function encapsulates the data retrieval logic, making it reusable.
    It's designed to handle larger, batched requests, which is more efficient
    than making many small requests. According to ECMWF guidelines, retrieving
    more data in a single request reduces the overhead of initiating
    multiple connections and helps the MARS system optimize tape access.

    For example, instead of looping and making a request for each day,
    provide a list of days in the 'day' field of the request dictionary.

    Args:
        dataset (str): The name of the dataset to retrieve from.
        request (dict): The dictionary containing the request parameters.
        target_file (str): The local path to save the downloaded file.
    """
    # Tip 1: Use a local file system.
    # We will check if the target directory exists and is writable.
    target_dir = os.path.dirname(target_file)
    if target_dir and not os.path.exists(target_dir):
        logging.info(f"Creating target directory: {target_dir}")
        os.makedirs(target_dir)

    # Simple check for writability
    try:
        # Use a more robust temporary file creation
        test_file_path = os.path.join(target_dir or '.', '.write_test')
        with open(test_file_path, 'w') as f:
            pass
        os.remove(test_file_path)
    except (IOError, OSError) as e:
        logging.error(f"Target directory '{target_dir or '.'}' is not writable: {e}")
        return

    logging.info("Initializing CDS API client.")
    client = cdsapi.Client()

    logging.info(f"Submitting data retrieval request for target '{target_file}'.")
    logging.info(f"Request details: {request}")

    # Tip 2 & 3: Estimate data volume.
    # While we can't perfectly estimate here, logging the request details
    # helps the user understand what is being downloaded.
    # For very large requests, consider adding a confirmation step.

    # Tip 4, 5, 6: Batch requests.
    # The `request` dictionary should be structured to retrieve as much data
    # as needed in a single go. For example, include multiple variables,
    # hours, or a range of days. This is what this script demonstrates.
    try:
        client.retrieve(dataset, request, target_file)
        logging.info(f"Successfully downloaded data to {target_file}")
    except Exception as e:
        logging.error(f"Failed to retrieve data: {e}")
        logging.error("Please check your request parameters and CDS API key setup (e.g., in ~/.cdsapirc).")
        # If the download fails, the cdsapi might leave an incomplete file.
        # It's good practice to clean it up.
        if os.path.exists(target_file):
            logging.warning(f"Removing potentially incomplete file: {target_file}")
            os.remove(target_file)


if __name__ == '__main__':
    # --- Example of an efficient request ---
    # This request downloads two variables, for all days in a month, for two times.
    # This is more efficient than making a separate request for each day or variable.

    # Reanalysis data is often stored together for a given month.
    # Requesting data for a full month at a time for all required variables
    # and levels is a good strategy.

    era5_dataset = 'reanalysis-era5-pressure-levels'

    # Tip 4: Batch related data into a single request.
    # Here we request geopotential and temperature for 5 days in March 2024
    # at two different times of the day and two pressure levels.
    efficient_request = {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            'geopotential', 'temperature',
        ],
        'pressure_level': [
            '500', '850',
        ],
        'year': '2024',
        'month': '03',
        'day': [
            '01', '02', '03', '04', '05',
        ],
        'time': [
            '00:00', '12:00',
        ],
        # Example of specifying a sub-region to reduce data volume.
        # This is highly recommended if you only need data for a specific area.
        # 'area': [60, -20, 50, 0], # North, West, South, East
    }

    # The target filename could reflect the content
    target_filename = 'era5_march_2024_5days_2vars_2levels.grib'

    print("--- Running efficient download example ---")
    download_era5_data(era5_dataset, efficient_request, target_filename)

    # --- Example of what to AVOID ---
    # The following shows an inefficient way of downloading the same data,
    # by making a request for each day individually. This leads to higher
    # server load and longer wait times for you.
    print("\n--- Inefficient download example (for demonstration) ---")
    print("The following approach is inefficient and should be avoided.")
    print("It would make 5 separate requests to the server instead of one.")

    # for day in ['01', '02', '03', '04', '05']:
    #     inefficient_request = {
    #         'product_type': 'reanalysis',
    #         'format': 'grib',
    #         'variable': ['geopotential', 'temperature'],
    #         'pressure_level': ['500', '850'],
    #         'year': '2024',
    #         'month': '03',
    #         'day': day,
    #         'time': ['00:00', '12:00'],
    #     }
    #     target = f'era5_2024_03_{day}.grib'
    #     # Calling download_era5_data here in a loop is inefficient.
    #     # download_era5_data(era5_dataset, inefficient_request, target)
    print("The inefficient code is commented out to prevent execution.") 