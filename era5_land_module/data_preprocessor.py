"""
ERA5_Land Data Preprocessor

This module provides functionality to preprocess ERA5_Land data including:
- Resampling to target resolution (e.g., 30m)
- Filling NaN values using interpolation
- Masking areas outside defined grids
- Converting to GeoTIFF format
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.fill import fillnodata
import xarray as xr
import pygrib

from .config import *
from .utils import (
    setup_logging, load_grid_data, get_grid_bboxes, ensure_cdo,
    parse_resolution_to_degrees, write_cdo_gridfile, cdo_remapbil_to_grid,
    cleanup_temp_files, save_metadata, format_file_size, create_progress_callback
)


class DataPreprocessor:
    """
    ERA5_Land data preprocessor for resampling, masking, and format conversion.
    
    This class handles preprocessing tasks including:
    - Resampling data to target resolution (e.g., 30m)
    - Filling NaN values using interpolation
    - Masking areas outside defined grids
    - Converting between GRIB, NetCDF, and GeoTIFF formats
    """
    
    def __init__(self,
                 target_resolution: str = '30m',
                 interpolation_method: str = 'bilinear',
                 fill_method: str = 'interpolate',
                 mask_outside_grids: bool = True,
                 output_format: str = 'tif',
                 compression: str = 'lzw',
                 nodata_value: float = -9999,
                 max_workers: int = 4):
        """
        Initialize data preprocessor.
        
        Args:
            target_resolution: Target resolution for resampling (e.g., '30m', '0.1deg')
            interpolation_method: Interpolation method ('bilinear', 'nearest', 'cubic')
            fill_method: Method for filling NaN values ('interpolate', 'fill', 'none')
            mask_outside_grids: Whether to mask areas outside defined grids
            output_format: Output format ('tif', 'grib', 'netcdf')
            compression: Compression method for output files
            nodata_value: NoData value for output files
            max_workers: Maximum number of parallel workers
        """
        self.logger = setup_logging('DataPreprocessor')
        self.target_resolution = target_resolution
        self.interpolation_method = interpolation_method
        self.fill_method = fill_method
        self.mask_outside_grids = mask_outside_grids
        self.output_format = output_format
        self.compression = compression
        self.nodata_value = nodata_value
        self.max_workers = max_workers
        
        # Parse target resolution
        self.target_resolution_deg = parse_resolution_to_degrees(target_resolution)
        
        # Load grid data if masking is enabled
        self.grid_data = None
        self.grid_bboxes = None
        if self.mask_outside_grids:
            try:
                self.grid_data = load_grid_data()
                self.grid_bboxes = get_grid_bboxes()
                self.logger.info(f"Loaded {len(self.grid_bboxes)} grid bounding boxes")
            except Exception as e:
                self.logger.warning(f"Failed to load grid data: {e}")
                self.mask_outside_grids = False
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_size': 0,
            'start_time': None,
            'end_time': None
        }
    
    def preprocess_directory(self,
                           input_dir: Path,
                           output_dir: Optional[Path] = None,
                           variables: Optional[List[str]] = None,
                           progress_callback: Optional[callable] = None) -> Dict:
        """
        Preprocess all files in a directory.
        
        Args:
            input_dir: Input directory containing ERA5_Land files
            output_dir: Output directory for processed files
            variables: List of variables to process (if None, processes all)
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary with processing statistics
        """
        self.logger.info(f"Starting preprocessing of directory: {input_dir}")
        self.logger.info(f"Target resolution: {self.target_resolution}")
        self.logger.info(f"Output format: {self.output_format}")
        
        # Reset statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_size': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Find input files
        input_files = self._find_input_files(input_dir, variables)
        self.stats['total_files'] = len(input_files)
        
        if not input_files:
            self.logger.warning("No input files found")
            return self.stats
        
        # Create output directory
        if output_dir is None:
            output_dir = PROCESSED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create processing tasks
        processing_tasks = self._create_processing_tasks(input_files, output_dir)
        
        # Execute processing
        self._execute_processing(processing_tasks, progress_callback)
        
        # Finalize statistics
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info("Preprocessing completed!")
        self.logger.info(f"Total files: {self.stats['total_files']}")
        self.logger.info(f"Processed: {self.stats['processed_files']}")
        self.logger.info(f"Failed: {self.stats['failed_files']}")
        self.logger.info(f"Total size: {format_file_size(self.stats['total_size'])}")
        self.logger.info(f"Duration: {duration}")
        
        return self.stats
    
    def preprocess_file(self,
                       input_file: Path,
                       output_file: Optional[Path] = None) -> Dict:
        """
        Preprocess a single file.
        
        Args:
            input_file: Input file path
            output_file: Output file path (if None, auto-generated)
            
        Returns:
            Dictionary with processing result
        """
        self.logger.info(f"Preprocessing file: {input_file}")
        
        try:
            # Create output file path if not provided
            if output_file is None:
                output_file = self._generate_output_path(input_file)
            
            # Create output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Process file
            result = self._process_single_file(input_file, output_file)
            
            if result['success']:
                self.logger.info(f"Successfully processed: {output_file}")
            else:
                self.logger.error(f"Failed to process: {input_file} - {result['error']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Exception during preprocessing: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'file_size': 0
            }
    
    def _find_input_files(self, input_dir: Path, variables: Optional[List[str]]) -> List[Path]:
        """
        Find input files in directory.
        
        Args:
            input_dir: Input directory
            variables: Variables to filter (if None, includes all)
            
        Returns:
            List of input file paths
        """
        # Find all GRIB and NetCDF files
        grib_files = list(input_dir.rglob("*.grib"))
        netcdf_files = list(input_dir.rglob("*.nc"))
        all_files = grib_files + netcdf_files
        
        # Filter by variables if specified
        if variables:
            filtered_files = []
            for file_path in all_files:
                file_variable = file_path.stem.split('_')[0]
                if file_variable in variables:
                    filtered_files.append(file_path)
            return filtered_files
        
        return all_files
    
    def _create_processing_tasks(self, input_files: List[Path], 
                                output_dir: Path) -> List[Dict]:
        """
        Create processing tasks for all input files.
        
        Args:
            input_files: List of input file paths
            output_dir: Output directory
            
        Returns:
            List of processing task dictionaries
        """
        tasks = []
        
        for input_file in input_files:
            output_file = self._generate_output_path(input_file, output_dir)
            
            task = {
                'input_file': input_file,
                'output_file': output_file,
                'variable': input_file.stem.split('_')[0]
            }
            
            tasks.append(task)
        
        return tasks
    
    def _execute_processing(self, tasks: List[Dict], 
                           progress_callback: Optional[callable]) -> None:
        """
        Execute processing tasks using parallel processing.
        
        Args:
            tasks: List of processing tasks
            progress_callback: Optional progress callback function
        """
        self.logger.info(f"Executing {len(tasks)} processing tasks with {self.max_workers} workers")
        
        # Create progress callback if not provided
        if progress_callback is None:
            progress_callback = create_progress_callback(len(tasks), "Preprocessing ERA5_Land data")
        
        # Execute processing in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_file, task['input_file'], task['output_file']): task 
                for task in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result['success']:
                        self.stats['processed_files'] += 1
                        self.stats['total_size'] += result['file_size']
                        self.logger.debug(f"Processed: {task['output_file'].name}")
                    else:
                        self.stats['failed_files'] += 1
                        self.logger.error(f"Failed: {task['input_file'].name} - {result['error']}")
                except Exception as e:
                    self.stats['failed_files'] += 1
                    self.logger.error(f"Exception in processing: {task['input_file'].name} - {e}")
                
                # Update progress
                progress_callback(1)
    
    def _process_single_file(self, input_file: Path, output_file: Path) -> Dict:
        """
        Process a single file.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            
        Returns:
            Dictionary with processing result
        """
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Step 1: Convert to intermediate format if needed
                intermediate_file = self._convert_to_intermediate(input_file, temp_path)
                
                # Step 2: Resample to target resolution
                resampled_file = self._resample_data(intermediate_file, temp_path)
                
                # Step 3: Fill NaN values
                filled_file = self._fill_nan_values(resampled_file, temp_path)
                
                # Step 4: Apply grid mask if enabled
                if self.mask_outside_grids:
                    masked_file = self._apply_grid_mask(filled_file, temp_path)
                else:
                    masked_file = filled_file
                
                # Step 5: Convert to final output format
                final_file = self._convert_to_output_format(masked_file, output_file)
                
                # Validate output file
                if final_file.exists() and final_file.stat().st_size > 0:
                    file_size = final_file.stat().st_size
                    
                    # Save processing metadata
                    metadata = self._create_processing_metadata(input_file, output_file, file_size)
                    metadata_path = output_file.with_suffix('.json')
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
                        'error': 'Output file validation failed'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'file_size': 0,
                'error': str(e)
            }
    
    def _convert_to_intermediate(self, input_file: Path, temp_dir: Path) -> Path:
        """
        Convert input file to intermediate format for processing.
        
        Args:
            input_file: Input file path
            temp_dir: Temporary directory
            
        Returns:
            Path to intermediate file
        """
        intermediate_file = temp_dir / f"intermediate_{input_file.stem}.nc"
        
        # Convert GRIB to NetCDF if needed
        if input_file.suffix.lower() in ['.grib', '.grb']:
            if ensure_cdo():
                # Use CDO to convert GRIB to NetCDF
                cmd = ['cdo', '-s', '-f', 'nc', 'copy', str(input_file), str(intermediate_file)]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                # Use pygrib and xarray
                with pygrib.open(str(input_file)) as grbs:
                    data = grbs.select()[0]
                    values = data.values
                    lats, lons = data.latlons()
                    
                    ds = xr.Dataset(
                        data_vars={
                            data.name: (['lat', 'lon'], values)
                        },
                        coords={
                            'lat': lats[:, 0],
                            'lon': lons[0, :]
                        }
                    )
                    ds.to_netcdf(intermediate_file)
        else:
            # Already NetCDF, just copy
            shutil.copy2(input_file, intermediate_file)
        
        return intermediate_file
    
    def _resample_data(self, input_file: Path, temp_dir: Path) -> Path:
        """
        Resample data to target resolution.
        
        Args:
            input_file: Input file path
            temp_dir: Temporary directory
            
        Returns:
            Path to resampled file
        """
        resampled_file = temp_dir / f"resampled_{input_file.stem}.nc"
        
        # Use CDO for resampling if available
        if ensure_cdo():
            # Create grid file for target resolution
            grid_file = temp_dir / "target_grid.txt"
            
            # Get bounds from input file
            with xr.open_dataset(input_file) as ds:
                lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
                lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
            
            # Write grid file
            write_cdo_gridfile(lon_min, lon_max, lat_min, lat_max, 
                             self.target_resolution_deg, grid_file)
            
            # Resample using CDO
            if cdo_remapbil_to_grid(input_file, resampled_file, grid_file):
                return resampled_file
        
        # Fallback to rasterio/xarray resampling
        with xr.open_dataset(input_file) as ds:
            # Get variable name
            var_name = list(ds.data_vars.keys())[0]
            
            # Create target coordinates
            lon_target = np.arange(ds.lon.min(), ds.lon.max(), self.target_resolution_deg)
            lat_target = np.arange(ds.lat.min(), ds.lat.max(), self.target_resolution_deg)
            
            # Resample
            ds_resampled = ds.interp(
                lon=lon_target,
                lat=lat_target,
                method=self.interpolation_method
            )
            
            # Save resampled data
            ds_resampled.to_netcdf(resampled_file)
        
        return resampled_file
    
    def _fill_nan_values(self, input_file: Path, temp_dir: Path) -> Path:
        """
        Fill NaN values in the data.
        
        Args:
            input_file: Input file path
            temp_dir: Temporary directory
            
        Returns:
            Path to file with filled values
        """
        if self.fill_method == 'none':
            return input_file
        
        filled_file = temp_dir / f"filled_{input_file.stem}.nc"
        
        with xr.open_dataset(input_file) as ds:
            # Get variable name
            var_name = list(ds.data_vars.keys())[0]
            
            # Fill NaN values
            if self.fill_method == 'interpolate':
                # Use xarray's interpolate_na
                ds_filled = ds.interpolate_na(dim='lon', method='linear')
                ds_filled = ds_filled.interpolate_na(dim='lat', method='linear')
            elif self.fill_method == 'fill':
                # Use forward fill then backward fill
                ds_filled = ds.fillna(method='ffill').fillna(method='bfill')
            else:
                ds_filled = ds
            
            # Save filled data
            ds_filled.to_netcdf(filled_file)
        
        return filled_file
    
    def _apply_grid_mask(self, input_file: Path, temp_dir: Path) -> Path:
        """
        Apply grid mask to exclude areas outside defined grids.
        
        Args:
            input_file: Input file path
            temp_dir: Temporary directory
            
        Returns:
            Path to masked file
        """
        if not self.mask_outside_grids or self.grid_data is None:
            return input_file
        
        masked_file = temp_dir / f"masked_{input_file.stem}.nc"
        
        with xr.open_dataset(input_file) as ds:
            # Get variable name
            var_name = list(ds.data_vars.keys())[0]
            
            # Create mask from grid geometries
            mask = self._create_grid_mask(ds.lon.values, ds.lat.values)
            
            # Apply mask
            ds_masked = ds.copy()
            ds_masked[var_name] = ds_masked[var_name].where(mask, self.nodata_value)
            
            # Save masked data
            ds_masked.to_netcdf(masked_file)
        
        return masked_file
    
    def _create_grid_mask(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """
        Create mask from grid geometries.
        
        Args:
            lons: Longitude coordinates
            lats: Latitude coordinates
            
        Returns:
            Boolean mask array
        """
        # Create coordinate grid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Initialize mask (False = outside grids)
        mask = np.zeros_like(lon_grid, dtype=bool)
        
        # Check each point against grid geometries
        for grid_id, bbox in self.grid_bboxes.items():
            # Simple bounding box check first
            bbox_mask = (
                (lon_grid >= bbox['west']) & (lon_grid <= bbox['east']) &
                (lat_grid >= bbox['south']) & (lat_grid <= bbox['north'])
            )
            mask |= bbox_mask
        
        return mask
    
    def _convert_to_output_format(self, input_file: Path, output_file: Path) -> Path:
        """
        Convert to final output format.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            
        Returns:
            Path to output file
        """
        if self.output_format.lower() == 'tif':
            return self._convert_to_geotiff(input_file, output_file)
        elif self.output_format.lower() in ['grib', 'grb']:
            return self._convert_to_grib(input_file, output_file)
        else:
            # NetCDF format
            shutil.copy2(input_file, output_file)
            return output_file
    
    def _convert_to_geotiff(self, input_file: Path, output_file: Path) -> Path:
        """
        Convert to GeoTIFF format.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            
        Returns:
            Path to GeoTIFF file
        """
        with xr.open_dataset(input_file) as ds:
            # Get variable name
            var_name = list(ds.data_vars.keys())[0]
            
            # Get data and coordinates
            data = ds[var_name].values
            lons = ds.lon.values
            lats = ds.lat.values
            
            # Create transform
            transform = from_bounds(
                lons.min(), lats.min(), lons.max(), lats.max(),
                data.shape[1], data.shape[0]
            )
            
            # Write GeoTIFF
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs='EPSG:4326',
                transform=transform,
                nodata=self.nodata_value,
                compress=self.compression
            ) as dst:
                dst.write(data, 1)
        
        return output_file
    
    def _convert_to_grib(self, input_file: Path, output_file: Path) -> Path:
        """
        Convert to GRIB format.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            
        Returns:
            Path to GRIB file
        """
        if ensure_cdo():
            # Use CDO to convert NetCDF to GRIB
            cmd = ['cdo', '-s', '-f', 'grb', 'copy', str(input_file), str(output_file)]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # Fallback: just copy if already GRIB
            shutil.copy2(input_file, output_file)
        
        return output_file
    
    def _generate_output_path(self, input_file: Path, output_dir: Optional[Path] = None) -> Path:
        """
        Generate output file path.
        
        Args:
            input_file: Input file path
            output_dir: Output directory (if None, uses default)
            
        Returns:
            Output file path
        """
        if output_dir is None:
            output_dir = PROCESSED_DIR
        
        # Create filename with processing info
        filename = f"{input_file.stem}_processed.{self.output_format}"
        return output_dir / filename
    
    def _create_processing_metadata(self, input_file: Path, output_file: Path, 
                                  file_size: int) -> Dict:
        """
        Create metadata for processed file.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            file_size: File size in bytes
            
        Returns:
            Metadata dictionary
        """
        return {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'processing_parameters': {
                'target_resolution': self.target_resolution,
                'interpolation_method': self.interpolation_method,
                'fill_method': self.fill_method,
                'mask_outside_grids': self.mask_outside_grids,
                'output_format': self.output_format,
                'compression': self.compression,
                'nodata_value': self.nodata_value
            },
            'file_size': file_size,
            'file_size_formatted': format_file_size(file_size),
            'processing_timestamp': datetime.now().isoformat(),
            'processing_info': {
                'preprocessor_version': '1.0.0',
                'target_resolution_degrees': self.target_resolution_deg
            }
        } 