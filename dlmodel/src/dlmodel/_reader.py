import os
import dask.array as da
from dask import delayed
import numpy as np
from openslide import OpenSlide, deepzoom
from typing import List, Tuple, Optional
from skimage.transform import resize
from csbdeep.utils import normalize
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_he')

# === A) Standalone rgb2hed ===
hed_from_rgb = np.array(
    [
        [1.87798274, -1.00767869, -0.55611582],
        [-0.06590806, 1.13473037, -0.1355218],
        [-0.60190736, -0.48041419, 1.57358807],
    ]
)

def separate_stains(rgb: np.ndarray, conv_matrix: np.ndarray) -> np.ndarray:
    np.maximum(rgb, 1e-6, out=rgb)  # avoid log(0)
    log_adjust = np.log(1e-6)
    stains = (np.log(rgb) / log_adjust) @ conv_matrix
    np.maximum(stains, 0, out=stains)
    return stains

# === C) Threshold mask from HED channel at full resolution ===
def stardist_predict(image: np.ndarray) -> np.ndarray:
    """Run StarDist prediction on a tile and return labeled mask"""
    if image.shape[0] < 64 or image.shape[1] < 64:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    image = normalize(image, 1, 99.8, axis=(0, 1))
    prob = model.predict_instances(image)[0] 
    # prob_resized = resize(prob, image.shape[:2], preserve_range=True).astype(np.uint8)
    return (prob * 255).astype(np.uint8)  # Napari-friendly label format

def _hed_threshold_mask(cropped: np.ndarray, thresh=0.05) -> np.ndarray:
    cropped_scaled = cropped / 255.0
    hed = separate_stains(cropped_scaled[:, :, :3], hed_from_rgb)
    mask = np.zeros((*cropped.shape[:2],), dtype=np.uint8)
    mask[hed[..., 0] > thresh] = 255  # label = 1 for hematoxylin
    return mask

def _get_deepzoom_tiles(svs_path: str, tile_size: int = 512):
    slide = OpenSlide(svs_path)
    dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)

    num_levels = dz.level_count
    pyramid_rgb = []

    for level in reversed(range(num_levels)):
        cols, rows = dz.level_tiles[level]
        level_shape = dz.level_dimensions[level]
        shape = (level_shape[1], level_shape[0], 3)

        tile_rows = []
        mask_rows = []
        label_rows = []
        for row in range(rows):
            tile_cols = []
            mask_cols = []
            label_cols = []
            for col in range(cols):
                tile = delayed(_load_tile)(dz, level, col, row, tile_size)
                tile_arr = da.from_delayed(tile, shape=(tile_size, tile_size, 3), dtype=np.uint8)
                tile_cols.append(tile_arr)
                
                if level == num_levels - 1:
                    mask_tile = delayed(_hed_threshold_mask)(tile)
                    mask_arr = da.from_delayed(mask_tile, shape=(tile_size, tile_size), dtype=np.uint8)
                    mask_cols.append(mask_arr)

                    labels_tile = delayed(stardist_predict)(tile)
                    labels_arr = da.from_delayed(labels_tile, shape=(tile_size, tile_size), dtype=np.uint8)
                    label_cols.append(labels_arr)

            row_arr = da.concatenate(tile_cols, axis=1)
            tile_rows.append(row_arr)

            if level == num_levels - 1:
                row_mask_arr = da.concatenate(mask_cols, axis=1)
                mask_rows.append(row_mask_arr)

                row_label_arr = da.concatenate(label_cols, axis=1)
                label_rows.append(row_label_arr)

        full_img = da.concatenate(tile_rows, axis=0)
        cropped = full_img[:shape[0], :shape[1], :]
        pyramid_rgb.append(cropped)

        if level == num_levels - 1:
            full_mask = da.concatenate(mask_rows, axis=0)
            full_mask = full_mask[:shape[0], :shape[1]]

            full_labels = da.concatenate(label_rows, axis=0)
            full_labels = full_labels[:shape[0], :shape[1]]


    add_kwargs_img = {
        "multiscale": True,
        "contrast_limits": [0, 255],
        "name": "H&E RGB",
    }
    add_kwargs_mask = {
        "name": "Hematoxylin mask",
        "opacity": 0.4,
        "contrast_limits": [0, 255],
        "blending": "additive"
    }

    add_kwargs_labels = {
        "name": "StarDist Nuclei",
        "opacity": 0.4,
        "blending": "additive",
        "contrast_limits": [0, 255],
    }

    return [
        (pyramid_rgb, add_kwargs_img, "image"),
        (full_mask, add_kwargs_mask, "image"),
        (full_labels, add_kwargs_labels, "image")
    ]


def _load_tile(dz: deepzoom.DeepZoomGenerator, level: int, col: int, row: int, tile_size: int) -> np.ndarray:
    tile = dz.get_tile(level, (col, row))
    tile_np = np.asarray(tile)
    if tile_np.shape[2] == 4:  # Convert RGBA → RGB
        tile_np = tile_np[:, :, :3]
    tile_reshape = resize_tile(tile_np, target_shape=(tile_size, tile_size, 3))
    return tile_reshape

def resize_tile(tile, target_shape):
    padded = np.zeros(target_shape, dtype=tile.dtype)
    cropped = tile[:target_shape[0], :target_shape[1], :]
    padded[:cropped.shape[0], :cropped.shape[1], :] = cropped
    return padded

def reader_function(path: str):
    if os.path.splitext(path)[-1].lower() != ".svs":
        return None
    return _get_deepzoom_tiles