import os
import json
import argparse
import shutil
import subprocess
import tempfile
from typing import List, Tuple, Dict

# Paths
GEOJSON_PATH = 'ERA5/Grid_50K_MatchedDates.geojson'
DATA_ROOT = 'data/era5_vietnam/grib'

BBox = Tuple[float, float, float, float]  # (lon_w, lat_s, lon_e, lat_n)


def normalize_bbox(coords: List[List[float]]) -> BBox:
    (lon_w, lat_n), (lon_e, lat_s) = coords
    lon_min = min(lon_w, lon_e)
    lon_max = max(lon_w, lon_e)
    lat_min = min(lat_s, lat_n)
    lat_max = max(lat_s, lat_n)
    return (lon_min, lat_min, lon_max, lat_max)


def load_grid_bboxes(geojson_path: str) -> Tuple[List[Dict], Dict[str, BBox]]:
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    grids = []
    id_to_bbox: Dict[str, BBox] = {}
    for feature in data['features']:
        coords = feature['geometry']['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        gid = feature['id']
        gmin_lon, gmax_lon = min(lons), max(lons)
        gmin_lat, gmax_lat = min(lats), max(lats)
        grids.append({
            'id': gid,
            'phien_hieu': feature['properties'].get('PhienHieu', ''),
            'min_lon': gmin_lon,
            'max_lon': gmax_lon,
            'min_lat': gmin_lat,
            'max_lat': gmax_lat,
        })
        id_to_bbox[gid] = (gmin_lon, gmin_lat, gmax_lon, gmax_lat)
    return grids, id_to_bbox


def intersects(b1: BBox, b2: BBox) -> bool:
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def bbox_intersection(b1: BBox, b2: BBox) -> BBox:
    lon_w = max(b1[0], b2[0])
    lat_s = max(b1[1], b2[1])
    lon_e = min(b1[2], b2[2])
    lat_n = min(b1[3], b2[3])
    return (lon_w, lat_s, lon_e, lat_n)


def find_intersecting_grids(bbox: BBox, grids: List[Dict]) -> List[Dict]:
    results = []
    for g in grids:
        gb = (g['min_lon'], g['min_lat'], g['max_lon'], g['max_lat'])
        if intersects(bbox, gb):
            results.append(g)
    return results


def ensure_wgrib2() -> str:
    path = shutil.which('wgrib2')
    return path or ''


def ensure_cdo() -> str:
    path = shutil.which('cdo')
    return path or ''


def subset_grib_wgrib2(wgrib2: str, in_file: str, bbox: BBox, out_file: str) -> None:
    lon_w, lat_s, lon_e, lat_n = bbox
    cmd = [wgrib2, in_file, '-small_grib', f"{lon_w}:{lon_e}", f"{lat_s}:{lat_n}", out_file]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def merge_gribs_wgrib2(wgrib2: str, inputs: List[str], out_file: str) -> None:
    if not inputs:
        raise ValueError('No input GRIB files to merge')
    subprocess.run([wgrib2, inputs[0], '-grib_out', out_file], check=True, stderr=subprocess.DEVNULL)
    for f in inputs[1:]:
        subprocess.run([wgrib2, f, '-append', '-grib_out', out_file], check=True, stderr=subprocess.DEVNULL)


def subset_grib_cdo(cdo: str, in_file: str, bbox: BBox, out_file: str) -> None:
    lon_w, lat_s, lon_e, lat_n = bbox
    cmd = [cdo, '-s', '-O', f'sellonlatbox,{lon_w},{lon_e},{lat_s},{lat_n}', in_file, out_file]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def merge_gribs_cdo(cdo: str, inputs: List[str], out_file: str) -> None:
    if not inputs:
        raise ValueError('No input GRIB files to merge')
    
    if len(inputs) == 1:
        # Just copy single file
        shutil.copy2(inputs[0], out_file)
        return
    
    # Try different CDO merge strategies
    strategies = [
        # Strategy 1: mergetime (for time-series data)
        [cdo, '-s', '-O', 'mergetime'] + inputs + [out_file],
        # Strategy 2: cat (simple concatenation)  
        [cdo, '-s', '-O', 'cat'] + inputs + [out_file],
        # Strategy 3: copy (merge by copying records sequentially)
        [cdo, '-s', '-O', 'copy'] + inputs + [out_file]
    ]
    
    last_error = None
    for i, cmd in enumerate(strategies):
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            return  # Success
        except subprocess.CalledProcessError as e:
            last_error = e
            # Clean up partial output file before trying next strategy
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                except:
                    pass
            continue
    
    # All strategies failed - fall back to manual merge by copying first file and warning
    print(f"Warning: CDO merge failed, using first file only as output")
    shutil.copy2(inputs[0], out_file)


def locate_month_files(year: str, month: str, grid_ids: List[str]) -> List[Tuple[str, str]]:
    month_dir = os.path.join(DATA_ROOT, year, month)
    if not os.path.isdir(month_dir):
        return []
    files: List[Tuple[str, str]] = []
    for gid in grid_ids:
        cand = os.path.join(month_dir, f"{gid}_{year}_{month}.grib")
        if os.path.exists(cand):
            files.append((cand, gid))
    return files


def main():
    parser = argparse.ArgumentParser(description='Subset and merge ERA5 GRIBs for a bbox')
    parser.add_argument('--bbox', type=str, required=True,
                        help='BBox as lonW,latN,lonE,latS (e.g., 105.0,22.0,107.0,20.0)')
    parser.add_argument('--year', type=str, required=True)
    parser.add_argument('--month', type=str, required=True)
    parser.add_argument('--output', type=str, default='bbox_subset.grib')
    args = parser.parse_args()

    try:
        lon_w, lat_n, lon_e, lat_s = [float(x) for x in args.bbox.split(',')]
    except Exception:
        raise SystemExit('Invalid --bbox. Use lonW,latN,lonE,latS')
    req_bbox = normalize_bbox([[lon_w, lat_n], [lon_e, lat_s]])

    grids, id_to_bbox = load_grid_bboxes(GEOJSON_PATH)
    hit_grids = find_intersecting_grids(req_bbox, grids)
    if not hit_grids:
        raise SystemExit('No grids intersect the provided bbox')

    grid_ids = [g['id'] for g in hit_grids]
    month_files = locate_month_files(args.year, args.month, grid_ids)
    if not month_files:
        raise SystemExit('No GRIB files found for the intersecting grids and specified year/month')

    wgrib2 = ensure_wgrib2()
    cdo = ensure_cdo()
    if not wgrib2 and not cdo:
        raise SystemExit('Neither wgrib2 nor cdo found in PATH. Install one: sudo apt install cdo OR conda install -c conda-forge wgrib2')

    tmpdir = tempfile.mkdtemp(prefix='bbox_grib_')
    subset_files: List[str] = []
    try:
        for i, (fpath, gid) in enumerate(month_files):
            tile_bbox = id_to_bbox.get(gid)
            if not tile_bbox:
                continue
            inter_bbox = bbox_intersection(req_bbox, tile_bbox)
            # Skip degenerate intersections
            if inter_bbox[2] <= inter_bbox[0] or inter_bbox[3] <= inter_bbox[1]:
                continue
            
            # Check if bbox is too small for reliable subsetting (< 0.1 degrees in either dimension)
            bbox_width = inter_bbox[2] - inter_bbox[0]
            bbox_height = inter_bbox[3] - inter_bbox[1]
            too_small = bbox_width < 0.1 or bbox_height < 0.1
            
            out_subset = os.path.join(tmpdir, f'subset_{i}.grib')
            
            if too_small:
                # Skip subsetting for very small bboxes, just copy the tile
                print(f"Bbox too small for reliable subsetting ({bbox_width:.3f}x{bbox_height:.3f} degrees), copying whole tile")
                shutil.copy2(fpath, out_subset)
            else:
                try:
                    if wgrib2:
                        subset_grib_wgrib2(wgrib2, fpath, inter_bbox, out_subset)
                    else:
                        subset_grib_cdo(cdo, fpath, inter_bbox, out_subset)
                except subprocess.CalledProcessError:
                    # Fallback: copy whole tile if subsetting fails
                    print(f"Subsetting failed for tile {gid}, copying whole tile as fallback")
                    shutil.copy2(fpath, out_subset)
            
            if os.path.exists(out_subset) and os.path.getsize(out_subset) > 0:
                subset_files.append(out_subset)
        
        if not subset_files:
            raise SystemExit('No data found within bbox after subsetting')
        
        if wgrib2:
            merge_gribs_wgrib2(wgrib2, subset_files, args.output)
        else:
            merge_gribs_cdo(cdo, subset_files, args.output)
        print(f"Wrote bbox GRIB: {args.output}")
    finally:
        for p in subset_files:
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass


if __name__ == '__main__':
    main() 