"""
ERA5_Land Data Searcher

This module provides functionality to search and extract ERA5_Land data for specific
regions and time periods from the downloaded database.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import xarray as xr
import pygrib

from .config import *
from .utils import (
    setup_logging, load_grid_data, get_grid_bboxes, check_intersection,
    parse_date_range, generate_date_list, save_metadata, format_file_size,
    create_progress_callback, calculate_bbox_area
)


class DataSearcher:
    """
    ERA5_Land data searcher for extracting data for specific regions and time periods.
    
    This class handles searching and extracting ERA5_Land data from the downloaded
    database for specific bounding boxes and time intervals.
    """
    
    def __init__(self,
                 data_dir: Optional[Path] = None,
                 max_bbox_size: float = 5.0,
                 time_buffer: int = 1,
                 output_resolution: str = '30m',
                 include_metadata: bool = True,
                 parallel_processing: bool = True,
                 max_workers: int = 4):
        """
        Initialize data searcher.
        
        Args:
            data_dir: Directory containing ERA5_Land data (if None, uses default)
            max_bbox_size: Maximum bounding box size in degrees
            time_buffer: Buffer in days around requested time period
            output_resolution: Output resolution for search results
            include_metadata: Whether to include metadata in output
            parallel_processing: Whether to use parallel processing
            max_workers: Maximum number of parallel workers
        """
        self.logger = setup_logging('DataSearcher')
        self.data_dir = data_dir if data_dir else DATA_DIR
        self.max_bbox_size = max_bbox_size
        self.time_buffer = time_buffer
        self.output_resolution = output_resolution
        self.include_metadata = include_metadata
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        
        # Load grid data for intersection checking
        try:
            self.grid_data = load_grid_data()
            self.grid_bboxes = get_grid_bboxes()
            self.logger.info(f"Loaded {len(self.grid_bboxes)} grid bounding boxes")
        except Exception as e:
            self.logger.warning(f"Failed to load grid data: {e}")
            self.grid_data = None
            self.grid_bboxes = None
        
        # Search statistics
        self.stats = {
            'total_files_found': 0,
            'files_processed': 0,
            'files_extracted': 0,
            'failed_extractions': 0,
            'total_size': 0,
            'start_time': None,
            'end_time': None
        }
    
    def search_data(self,
                   bbox: Tuple[Tuple[float, float], Tuple[float, float]],
                   start_date: str,
                   end_date: str,
                   variables: Optional[List[str]] = None,
                   output_dir: Optional[Path] = None,
                   progress_callback: Optional[callable] = None) -> Dict:
        """
        Search and extract ERA5_Land data for the specified region and time period.
        
        Args:
            bbox: Bounding box as ((top_left_lon, top_left_lat), (bottom_right_lon, bottom_right_lat))
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            variables: List of variables to extract (if None, extracts all available)
            output_dir: Output directory for extracted data
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary with search results and statistics
        """
        self.logger.info(f"Starting data search for bbox: {bbox}")
        self.logger.info(f"Time period: {start_date} to {end_date}")
        
        # Reset statistics
        self.stats = {
            'total_files_found': 0,
            'files_processed': 0,
            'files_extracted': 0,
            'failed_extractions': 0,
            'total_size': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
        
        # Validate and normalize bounding box
        normalized_bbox = self._normalize_bbox(bbox)
        
        # Validate bounding box size
        bbox_area = calculate_bbox_area(normalized_bbox)
        if bbox_area > self.max_bbox_size * self.max_bbox_size:
            raise ValueError(f"Bounding box too large: {bbox_area:.2f} sq degrees (max: {self.max_bbox_size * self.max_bbox_size})")
        
        # Parse date range
        start_dt, end_dt = parse_date_range(start_date, end_date)
        
        # Add time buffer
        start_dt_buffered = start_dt - timedelta(days=self.time_buffer)
        end_dt_buffered = end_dt + timedelta(days=self.time_buffer)
        
        # Find relevant files
        relevant_files = self._find_relevant_files(normalized_bbox, start_dt_buffered, end_dt_buffered, variables)
        self.stats['total_files_found'] = len(relevant_files)
        
        if not relevant_files:
            self.logger.warning("No relevant files found")
            return self.stats
        
        # Create output directory
        if output_dir is None:
            output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create extraction tasks
        extraction_tasks = self._create_extraction_tasks(relevant_files, normalized_bbox, output_dir)
        
        # Execute extractions
        self._execute_extractions(extraction_tasks, progress_callback)
        
        # Finalize statistics
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info("Data search completed!")
        self.logger.info(f"Files found: {self.stats['total_files_found']}")
        self.logger.info(f"Files processed: {self.stats['files_processed']}")
        self.logger.info(f"Files extracted: {self.stats['files_extracted']}")
        self.logger.info(f"Failed extractions: {self.stats['failed_extractions']}")
        self.logger.info(f"Total size: {format_file_size(self.stats['total_size'])}")
        self.logger.info(f"Duration: {duration}")
        
        return self.stats
    
    def search_variable(self,
                       variable: str,
                       bbox: Tuple[Tuple[float, float], Tuple[float, float]],
                       start_date: str,
                       end_date: str,
                       output_dir: Optional[Path] = None) -> Dict:
        """
        Search and extract a single variable for the specified region and time period.
        
        Args:
            variable: Variable name to extract
            bbox: Bounding box
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_dir: Output directory
            
        Returns:
            Dictionary with search results
        """
        return self.search_data(bbox, start_date, end_date, [variable], output_dir)
    
    def _normalize_bbox(self, bbox: Tuple[Tuple[float, float], Tuple[float, float]]) -> Dict[str, float]:
        """
        Normalize bounding box to standard format.
        
        Args:
            bbox: Bounding box as ((top_left_lon, top_left_lat), (bottom_right_lon, bottom_right_lat))
            
        Returns:
            Normalized bounding box dictionary
        """
        (top_left_lon, top_left_lat), (bottom_right_lon, bottom_right_lat) = bbox
        
        return {
            'west': min(top_left_lon, bottom_right_lon),
            'east': max(top_left_lon, bottom_right_lon),
            'north': max(top_left_lat, bottom_right_lat),
            'south': min(top_left_lat, bottom_right_lat)
        }
    
    def _find_relevant_files(self, bbox: Dict[str, float], start_date: datetime, 
                           end_date: datetime, variables: Optional[List[str]]) -> List[Dict]:
        """
        Find relevant files for the specified region and time period.
        
        Args:
            bbox: Normalized bounding box
            start_date: Start date
            end_date: End date
            variables: Variables to search for
            
        Returns:
            List of relevant file dictionaries
        """
        relevant_files = []
        
        # Search in both GRIB and NetCDF directories
        search_dirs = [GRIB_DIR, NETCDF_DIR]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            # Find all files in the directory
            for file_path in search_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.grib', '.grb', '.nc']:
                    file_info = self._extract_file_info(file_path)
                    
                    if file_info is None:
                        continue
                    
                    # Check if file is relevant
                    if self._is_file_relevant(file_info, bbox, start_date, end_date, variables):
                        relevant_files.append(file_info)
        
        return relevant_files
    
    def _extract_file_info(self, file_path: Path) -> Optional[Dict]:
        """
        Extract information from file path and metadata.
        
        Args:
            file_path: File path
            
        Returns:
            File information dictionary or None if extraction fails
        """
        try:
            # Parse filename: variable_year_month_day_hour.format
            filename_parts = file_path.stem.split('_')
            
            if len(filename_parts) < 5:
                return None
            
            variable = filename_parts[0]
            year = int(filename_parts[1])
            month = int(filename_parts[2])
            day = int(filename_parts[3])
            hour = int(filename_parts[4])
            
            # Create datetime
            file_date = datetime(year, month, day, hour)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Try to load metadata
            metadata_path = file_path.with_suffix('.json')
            metadata = None
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            return {
                'path': file_path,
                'variable': variable,
                'date': file_date,
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'file_size': file_size,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to extract file info from {file_path}: {e}")
            return None
    
    def _is_file_relevant(self, file_info: Dict, bbox: Dict[str, float], 
                         start_date: datetime, end_date: datetime, 
                         variables: Optional[List[str]]) -> bool:
        """
        Check if file is relevant for the search criteria.
        
        Args:
            file_info: File information dictionary
            bbox: Bounding box
            start_date: Start date
            end_date: End date
            variables: Variables to search for
            
        Returns:
            True if file is relevant
        """
        # Check variable
        if variables and file_info['variable'] not in variables:
            return False
        
        # Check date range
        if not (start_date <= file_info['date'] <= end_date):
            return False
        
        # Check spatial intersection if metadata is available
        if file_info['metadata'] and 'bbox' in file_info['metadata']:
            file_bbox = file_info['metadata']['bbox']
            if not check_intersection(bbox, file_bbox):
                return False
        
        return True
    
    def _create_extraction_tasks(self, relevant_files: List[Dict], bbox: Dict[str, float], 
                                output_dir: Path) -> List[Dict]:
        """
        Create extraction tasks for relevant files.
        
        Args:
            relevant_files: List of relevant file dictionaries
            bbox: Bounding box
            output_dir: Output directory
            
        Returns:
            List of extraction task dictionaries
        """
        tasks = []
        
        for file_info in relevant_files:
            # Create output filename
            output_filename = f"{file_info['variable']}_{bbox['west']:.3f}_{bbox['south']:.3f}_{bbox['east']:.3f}_{bbox['north']:.3f}_{file_info['date'].strftime('%Y%m%d_%H%M')}.tif"
            output_path = output_dir / output_filename
            
            task = {
                'file_info': file_info,
                'bbox': bbox,
                'output_path': output_path
            }
            
            tasks.append(task)
        
        return tasks
    
    def _execute_extractions(self, tasks: List[Dict], 
                           progress_callback: Optional[callable]) -> None:
        """
        Execute extractions using parallel processing.
        
        Args:
            tasks: List of extraction tasks
            progress_callback: Optional progress callback function
        """
        self.logger.info(f"Executing {len(tasks)} extraction tasks")
        
        # Create progress callback if not provided
        if progress_callback is None:
            progress_callback = create_progress_callback(len(tasks), "Extracting ERA5_Land data")
        
        if self.parallel_processing:
            # Execute extractions in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._extract_single_file, task): task 
                    for task in tasks
                }
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        self._update_stats(result)
                    except Exception as e:
                        self.stats['failed_extractions'] += 1
                        self.logger.error(f"Exception in extraction: {task['file_info']['path']} - {e}")
                    
                    # Update progress
                    progress_callback(1)
        else:
            # Execute extractions sequentially
            for task in tasks:
                try:
                    result = self._extract_single_file(task)
                    self._update_stats(result)
                except Exception as e:
                    self.stats['failed_extractions'] += 1
                    self.logger.error(f"Exception in extraction: {task['file_info']['path']} - {e}")
                
                # Update progress
                progress_callback(1)
    
    def _extract_single_file(self, task: Dict) -> Dict:
        """
        Extract data from a single file for the specified bounding box.
        
        Args:
            task: Extraction task dictionary
            
        Returns:
            Dictionary with extraction result
        """
        file_info = task['file_info']
        bbox = task['bbox']
        output_path = task['output_path']
        
        self.stats['files_processed'] += 1
        
        try:
            # Extract data based on file format
            if file_info['path'].suffix.lower() in ['.grib', '.grb']:
                result = self._extract_from_grib(file_info, bbox, output_path)
            else:
                result = self._extract_from_netcdf(file_info, bbox, output_path)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'file_size': 0,
                'error': str(e)
            }
    
    def _extract_from_grib(self, file_info: Dict, bbox: Dict[str, float], 
                          output_path: Path) -> Dict:
        """
        Extract data from GRIB file.
        
        Args:
            file_info: File information dictionary
            bbox: Bounding box
            output_path: Output file path
            
        Returns:
            Dictionary with extraction result
        """
        try:
            # Open GRIB file
            with pygrib.open(str(file_info['path'])) as grbs:
                # Select the first message (assuming single variable)
                grb = grbs.select()[0]
                
                # Get data and coordinates
                data = grb.values
                lats, lons = grb.latlons()
                
                # Create mask for bounding box
                mask = (
                    (lons >= bbox['west']) & (lons <= bbox['east']) &
                    (lats >= bbox['south']) & (lats <= bbox['north'])
                )
                
                # Extract subset
                lat_indices = np.where(np.any(mask, axis=1))[0]
                lon_indices = np.where(np.any(mask, axis=0))[0]
                
                if len(lat_indices) == 0 or len(lon_indices) == 0:
                    return {
                        'success': False,
                        'file_size': 0,
                        'error': 'No data in bounding box'
                    }
                
                # Extract data subset
                data_subset = data[lat_indices[0]:lat_indices[-1]+1, 
                                 lon_indices[0]:lon_indices[-1]+1]
                lats_subset = lats[lat_indices[0]:lat_indices[-1]+1, 
                                 lon_indices[0]:lon_indices[-1]+1]
                lons_subset = lons[lat_indices[0]:lat_indices[-1]+1, 
                                 lon_indices[0]:lon_indices[-1]+1]
                
                # Create transform
                transform = from_bounds(
                    lons_subset.min(), lats_subset.min(),
                    lons_subset.max(), lats_subset.max(),
                    data_subset.shape[1], data_subset.shape[0]
                )
                
                # Write GeoTIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=data_subset.shape[0],
                    width=data_subset.shape[1],
                    count=1,
                    dtype=data_subset.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                    nodata=-9999,
                    compress='lzw'
                ) as dst:
                    dst.write(data_subset, 1)
                
                # Save metadata if enabled
                if self.include_metadata:
                    metadata = self._create_extraction_metadata(file_info, bbox, output_path)
                    metadata_path = output_path.with_suffix('.json')
                    save_metadata(metadata, metadata_path)
                
                file_size = output_path.stat().st_size
                
                return {
                    'success': True,
                    'file_size': file_size,
                    'error': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'file_size': 0,
                'error': str(e)
            }
    
    def _extract_from_netcdf(self, file_info: Dict, bbox: Dict[str, float], 
                            output_path: Path) -> Dict:
        """
        Extract data from NetCDF file.
        
        Args:
            file_info: File information dictionary
            bbox: Bounding box
            output_path: Output file path
            
        Returns:
            Dictionary with extraction result
        """
        try:
            # Open NetCDF file
            with xr.open_dataset(file_info['path']) as ds:
                # Get variable name
                var_name = list(ds.data_vars.keys())[0]
                
                # Select data within bounding box
                ds_subset = ds.sel(
                    lon=slice(bbox['west'], bbox['east']),
                    lat=slice(bbox['south'], bbox['north'])
                )
                
                if ds_subset[var_name].size == 0:
                    return {
                        'success': False,
                        'file_size': 0,
                        'error': 'No data in bounding box'
                    }
                
                # Get data and coordinates
                data = ds_subset[var_name].values
                lons = ds_subset.lon.values
                lats = ds_subset.lat.values
                
                # Create transform
                transform = from_bounds(
                    lons.min(), lats.min(), lons.max(), lats.max(),
                    data.shape[1], data.shape[0]
                )
                
                # Write GeoTIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                    nodata=-9999,
                    compress='lzw'
                ) as dst:
                    dst.write(data, 1)
                
                # Save metadata if enabled
                if self.include_metadata:
                    metadata = self._create_extraction_metadata(file_info, bbox, output_path)
                    metadata_path = output_path.with_suffix('.json')
                    save_metadata(metadata, metadata_path)
                
                file_size = output_path.stat().st_size
                
                return {
                    'success': True,
                    'file_size': file_size,
                    'error': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'file_size': 0,
                'error': str(e)
            }
    
    def _update_stats(self, result: Dict) -> None:
        """
        Update statistics based on extraction result.
        
        Args:
            result: Extraction result dictionary
        """
        if result['success']:
            self.stats['files_extracted'] += 1
            self.stats['total_size'] += result['file_size']
        else:
            self.stats['failed_extractions'] += 1
    
    def _create_extraction_metadata(self, file_info: Dict, bbox: Dict[str, float], 
                                  output_path: Path) -> Dict:
        """
        Create metadata for extracted file.
        
        Args:
            file_info: File information dictionary
            bbox: Bounding box
            output_path: Output file path
            
        Returns:
            Metadata dictionary
        """
        return {
            'original_file': str(file_info['path']),
            'extracted_file': str(output_path),
            'variable': file_info['variable'],
            'date': file_info['date'].isoformat(),
            'bbox': bbox,
            'bbox_area': calculate_bbox_area(bbox),
            'extraction_timestamp': datetime.now().isoformat(),
            'extraction_info': {
                'searcher_version': '1.0.0',
                'output_resolution': self.output_resolution,
                'include_metadata': self.include_metadata
            }
        }
    
    def get_search_status(self, output_dir: Path) -> Dict:
        """
        Get status of extracted files.
        
        Args:
            output_dir: Output directory to check
            
        Returns:
            Dictionary with search status information
        """
        if not output_dir.exists():
            return {'error': 'Output directory does not exist'}
        
        files = list(output_dir.glob("*.tif"))
        metadata_files = list(output_dir.glob("*.json"))
        
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
            'output_format': 'tif',
            'output_resolution': self.output_resolution
        } 