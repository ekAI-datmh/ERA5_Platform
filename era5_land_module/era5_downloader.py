"""
ERA5_Land Data Downloader

This module provides functionality to download ERA5_Land data from the Climate Data Store (CDS)
for Vietnam and other regions.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi
import xarray as xr
import numpy as np

try:
    from .config import *
    from .utils import (
        setup_logging, create_cds_client, parse_date_range, generate_date_list,
        create_storage_paths, validate_file, save_metadata, format_file_size,
        estimate_download_size, create_progress_callback
    )
except ImportError:
    from config import *
    from utils import (
        setup_logging, create_cds_client, parse_date_range, generate_date_list,
        create_storage_paths, validate_file, save_metadata, format_file_size,
        estimate_download_size, create_progress_callback
    )


class ERA5Downloader:
    """
    ERA5_Land data downloader for Vietnam and other regions.
    
    This class handles downloading ERA5_Land data from the Climate Data Store (CDS)
    with support for both hourly and daily data, multiple variables, and efficient
    storage organization.
    """
    
    def __init__(self, 
                 dataset_type: str = 'hourly',
                 format_type: str = 'grib',
                 variables: Optional[List[str]] = None,
                 bbox: Optional[Dict[str, float]] = None,
                 max_workers: int = 4,
                 retry_attempts: int = 3,
                 delay_between_requests: float = 1.0):
        """
        Initialize ERA5_Land downloader.
        
        Args:
            dataset_type: Type of dataset ('hourly' or 'daily')
            format_type: Output format ('grib' or 'netcdf')
            variables: List of variables to download (if None, uses default list)
            bbox: Bounding box for download area (if None, uses Vietnam bbox)
            max_workers: Maximum number of parallel download workers
            retry_attempts: Number of retry attempts for failed downloads
            delay_between_requests: Delay between requests in seconds
        """
        self.logger = setup_logging('ERA5Downloader')
        self.dataset_type = dataset_type
        self.format_type = format_type
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.delay_between_requests = delay_between_requests
        
        # Set variables
        if variables is None:
            if dataset_type == 'hourly':
                self.variables = ERA5_LAND_HOURLY_VARIABLES
            else:
                self.variables = ERA5_LAND_DAILY_VARIABLES
        else:
            self.variables = variables
        
        # Set bounding box
        self.bbox = bbox if bbox else VIETNAM_BBOX
        
        # Initialize CDS client
        try:
            self.cds_client = create_cds_client()
            self.logger.info("CDS client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize CDS client: {e}")
            raise
        
        # Thread lock for logging
        self.log_lock = threading.Lock()
        
        # Download statistics
        self.stats = {
            'total_requests': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size': 0,
            'start_time': None,
            'end_time': None
        }
    
    def download_data(self, 
                     start_date: str,
                     end_date: str,
                     output_dir: Optional[Path] = None,
                     progress_callback: Optional[callable] = None) -> Dict:
        """
        Download ERA5_Land data for the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_dir: Output directory (if None, uses default structure)
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary with download statistics
        """
        self.logger.info(f"Starting ERA5_Land download: {start_date} to {end_date}")
        self.logger.info(f"Dataset: {self.dataset_type}, Format: {self.format_type}")
        self.logger.info(f"Variables: {len(self.variables)} variables")
        self.logger.info(f"Bounding box: {self.bbox}")
        
        # Reset statistics
        self.stats = {
            'total_requests': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
        
        # Parse date range
        start_dt, end_dt = parse_date_range(start_date, end_date)
        
        # Generate date list
        dates = generate_date_list(start_dt, end_dt, self.dataset_type)
        
        # Estimate download size
        estimated_size = estimate_download_size(
            self.bbox, self.variables, start_dt, end_dt, self.format_type
        )
        self.logger.info(f"Estimated download size: {format_file_size(estimated_size)}")
        
        # Create download tasks
        download_tasks = self._create_download_tasks(dates, output_dir)
        
        # Execute downloads
        self._execute_downloads(download_tasks, progress_callback)
        
        # Finalize statistics
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info("Download completed!")
        self.logger.info(f"Total requests: {self.stats['total_requests']}")
        self.logger.info(f"Successful: {self.stats['successful_downloads']}")
        self.logger.info(f"Failed: {self.stats['failed_downloads']}")
        self.logger.info(f"Total size: {format_file_size(self.stats['total_size'])}")
        self.logger.info(f"Duration: {duration}")
        
        return self.stats
    
    def _create_download_tasks(self, dates: List[datetime], 
                              output_dir: Optional[Path]) -> List[Dict]:
        """
        Create download tasks for all variables and dates.
        
        Args:
            dates: List of dates to download
            output_dir: Output directory
            
        Returns:
            List of download task dictionaries
        """
        tasks = []
        
        for date in dates:
            year = date.year
            month = date.month
            day = date.day
            hour = date.hour if self.dataset_type == 'hourly' else 0
            
            # Create storage paths
            paths = create_storage_paths(year, month, self.format_type)
            
            for variable in self.variables:
                # Create output file path
                if output_dir:
                    file_path = output_dir / f"{variable}_{year}_{month:02d}_{day:02d}_{hour:02d}.{self.format_type}"
                else:
                    file_path = paths['month'] / f"{variable}_{year}_{month:02d}_{day:02d}_{hour:02d}.{self.format_type}"
                
                # Create task
                task = {
                    'variable': variable,
                    'date': date,
                    'year': year,
                    'month': month,
                    'day': day,
                    'hour': hour,
                    'output_path': file_path,
                    'bbox': self.bbox.copy()
                }
                
                tasks.append(task)
        
        return tasks
    
    def _execute_downloads(self, tasks: List[Dict], 
                          progress_callback: Optional[callable]) -> None:
        """
        Execute downloads using parallel processing.
        
        Args:
            tasks: List of download tasks
            progress_callback: Optional progress callback function
        """
        self.logger.info(f"Executing {len(tasks)} download tasks with {self.max_workers} workers")
        
        # Create progress callback if not provided
        if progress_callback is None:
            progress_callback = create_progress_callback(len(tasks), "Downloading ERA5_Land data")
        
        # Execute downloads in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._download_single_task, task): task 
                for task in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result['success']:
                        self.stats['successful_downloads'] += 1
                        self.stats['total_size'] += result['file_size']
                        with self.log_lock:
                            self.logger.debug(f"Downloaded: {task['output_path'].name}")
                    else:
                        self.stats['failed_downloads'] += 1
                        with self.log_lock:
                            self.logger.error(f"Failed: {task['output_path'].name} - {result['error']}")
                except Exception as e:
                    self.stats['failed_downloads'] += 1
                    with self.log_lock:
                        self.logger.error(f"Exception in download: {task['output_path'].name} - {e}")
                
                # Update progress
                progress_callback(1)
                
                # Add delay between requests
                time.sleep(self.delay_between_requests)
    
    def _download_single_task(self, task: Dict) -> Dict:
        """
        Download a single task.
        
        Args:
            task: Download task dictionary
            
        Returns:
            Dictionary with download result
        """
        self.stats['total_requests'] += 1
        
        try:
            # Prepare CDS request parameters
            request_params = self._prepare_request_params(task)
            
            # Download file
            self.cds_client.retrieve(
                ERA5_LAND_DATASET,
                request_params,
                str(task['output_path'])
            )
            
            # Validate downloaded file
            if validate_file(task['output_path']):
                file_size = task['output_path'].stat().st_size
                
                # Save metadata
                metadata = self._create_metadata(task, file_size)
                metadata_path = task['output_path'].with_suffix('.json')
                save_metadata(metadata, metadata_path)
                
                return {
                    'success': True,
                    'file_size': file_size,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'file_size': 0,
                    'error': 'File validation failed'
                }
                
        except Exception as e:
            # Clean up failed file
            if task['output_path'].exists():
                task['output_path'].unlink()
            
            return {
                'success': False,
                'file_size': 0,
                'error': str(e)
            }
    
    def _prepare_request_params(self, task: Dict) -> Dict:
        """
        Prepare CDS API request parameters.
        
        Args:
            task: Download task dictionary
            
        Returns:
            CDS API request parameters
        """
        params = {
            'variable': task['variable'],
            'year': str(task['year']),
            'month': f"{task['month']:02d}",
            'day': f"{task['day']:02d}",
            'area': [
                task['bbox']['north'],
                task['bbox']['west'],
                task['bbox']['south'],
                task['bbox']['east']
            ],
            'format': self.format_type
        }
        
        # Add time parameter for hourly data
        if self.dataset_type == 'hourly':
            params['time'] = [f"{task['hour']:02d}:00"]
        
        return params
    
    def _create_metadata(self, task: Dict, file_size: int) -> Dict:
        """
        Create metadata for downloaded file.
        
        Args:
            task: Download task dictionary
            file_size: File size in bytes
            
        Returns:
            Metadata dictionary
        """
        return {
            'variable': task['variable'],
            'date': task['date'].isoformat(),
            'year': task['year'],
            'month': task['month'],
            'day': task['day'],
            'hour': task['hour'],
            'dataset_type': self.dataset_type,
            'format_type': self.format_type,
            'bbox': task['bbox'],
            'file_size': file_size,
            'file_size_formatted': format_file_size(file_size),
            'download_timestamp': datetime.now().isoformat(),
            'processing_info': {
                'downloader_version': '1.0.0',
                'cds_dataset': ERA5_LAND_DATASET
            }
        }
    
    def download_variable(self, 
                         variable: str,
                         start_date: str,
                         end_date: str,
                         output_dir: Optional[Path] = None) -> Dict:
        """
        Download a single variable for the specified date range.
        
        Args:
            variable: Variable name to download
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_dir: Output directory
            
        Returns:
            Dictionary with download statistics
        """
        # Temporarily set single variable
        original_variables = self.variables
        self.variables = [variable]
        
        try:
            result = self.download_data(start_date, end_date, output_dir)
        finally:
            # Restore original variables
            self.variables = original_variables
        
        return result
    
    def download_month(self, 
                      year: int,
                      month: int,
                      variables: Optional[List[str]] = None,
                      output_dir: Optional[Path] = None) -> Dict:
        """
        Download data for a specific month.
        
        Args:
            year: Year
            month: Month (1-12)
            variables: Variables to download (if None, uses all variables)
            output_dir: Output directory
            
        Returns:
            Dictionary with download statistics
        """
        start_date = f"{year}-{month:02d}-01"
        
        # Calculate end date (last day of month)
        if month == 12:
            end_date = f"{year}-12-31"
        else:
            next_month = datetime(year, month + 1, 1)
            end_date = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Set variables if specified
        if variables:
            original_variables = self.variables
            self.variables = variables
        
        try:
            result = self.download_data(start_date, end_date, output_dir)
        finally:
            # Restore original variables if changed
            if variables:
                self.variables = original_variables
        
        return result
    
    def get_download_status(self, output_dir: Path) -> Dict:
        """
        Get status of downloaded files.
        
        Args:
            output_dir: Directory to check
            
        Returns:
            Dictionary with download status information
        """
        if not output_dir.exists():
            return {'error': 'Output directory does not exist'}
        
        files = list(output_dir.rglob(f"*.{self.format_type}"))
        metadata_files = list(output_dir.rglob("*.json"))
        
        total_size = sum(f.stat().st_size for f in files)
        
        # Group files by variable
        variables = {}
        for file_path in files:
            variable = file_path.stem.split('_')[0]
            if variable not in variables:
                variables[variable] = []
            variables[variable].append({
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
        
        return {
            'total_files': len(files),
            'total_metadata_files': len(metadata_files),
            'total_size': total_size,
            'total_size_formatted': format_file_size(total_size),
            'variables': variables,
            'format_type': self.format_type,
            'dataset_type': self.dataset_type
        }
    
    def validate_downloads(self, output_dir: Path) -> Dict:
        """
        Validate downloaded files.
        
        Args:
            output_dir: Directory containing downloaded files
            
        Returns:
            Dictionary with validation results
        """
        if not output_dir.exists():
            return {'error': 'Output directory does not exist'}
        
        files = list(output_dir.rglob(f"*.{self.format_type}"))
        
        validation_results = {
            'total_files': len(files),
            'valid_files': 0,
            'invalid_files': 0,
            'errors': []
        }
        
        for file_path in files:
            try:
                # Check file size
                if file_path.stat().st_size < 1000:
                    validation_results['invalid_files'] += 1
                    validation_results['errors'].append(f"File too small: {file_path}")
                    continue
                
                # Try to open file
                if self.format_type == 'grib':
                    import pygrib
                    with pygrib.open(str(file_path)) as grbs:
                        if grbs.messagenumber == 0:
                            validation_results['invalid_files'] += 1
                            validation_results['errors'].append(f"Empty GRIB file: {file_path}")
                            continue
                elif self.format_type == 'netcdf':
                    with xr.open_dataset(file_path) as ds:
                        if len(ds.data_vars) == 0:
                            validation_results['invalid_files'] += 1
                            validation_results['errors'].append(f"Empty NetCDF file: {file_path}")
                            continue
                
                validation_results['valid_files'] += 1
                
            except Exception as e:
                validation_results['invalid_files'] += 1
                validation_results['errors'].append(f"Error validating {file_path}: {e}")
        
        return validation_results 