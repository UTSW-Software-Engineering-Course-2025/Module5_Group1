"""
This module is an example of a barebones lazy reader plugin for napari
for reading whole-slide image formats supported by openslide, using dask
for lazy evaluation.

It implements the Reader specification:
https://napari.org/stable/plugins/guides.html#readers
"""

import os

import dask.array as da
import numpy as np
from dask import delayed
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # A path list is not supported, so we return None.
        return None

    # We use a try-except block to gracefully handle files that
    # openslide cannot read, even if they have a supported extension.
    try:
        # Check if openslide can open the file.
        open_slide(path)
    except Exception:
        # If openslide fails, it's not a file we can read.
        return None

    # If the file is readable by openslide, return the reader function.
    return reader_function


def reader_function(path):
    """
    Take a path and return a list of LayerData tuples for a multiscale pyramid.

    Parameters
    ----------
    path : str
        Path to a whole-slide image file.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a dask array for each
        level of the image pyramid.
    """
    # --- 1. Setup: Open the slide and create a Deep Zoom generator ---
    slide = open_slide(path)
    TILE_SIZE = 256
    OVERLAP = 0
    generator = DeepZoomGenerator(
        slide, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=False
    )
    num_levels = generator.level_count

    # --- 2. Create a lazy function to get tiles ---
    @delayed(pure=True)
    def get_tile(level, col, row):
        """Delayed function to get a tile and convert it to a NumPy array."""
        tile = generator.get_tile(level, (col, row))
        return np.array(tile)[:, :, :3]

    # --- 3. Build a Dask array for each pyramid level ---
    # Get the dtype from a single sample tile before the loops
    try:
        sample_dtype = get_tile(num_levels - 1, 0, 0).compute().dtype
    except Exception:
        # Fallback if the highest-resolution level is empty or problematic
        sample_dtype = np.uint8

    pyramid = []
    # Loop from lowest resolution (smallest image) to highest (largest image)
    for level in reversed(range(num_levels)):
        num_tiles_x, num_tiles_y = generator.level_tiles[level]

        # Create a list of row arrays. Each row is built by concatenating its tiles.
        rows_of_arrays = []
        for r in range(num_tiles_y):
            row_of_tiles = []
            for c in range(num_tiles_x):
                # Use get_tile_dimensions to find the correct shape for each tile,
                # as edge tiles will not be the full TILE_SIZE.
                tile_w, tile_h = generator.get_tile_dimensions(level, (c, r))
                tile_shape = (tile_h, tile_w, 3)  # (rows, cols, channels)

                # Create the delayed dask array with the correct, specific shape
                dask_tile = da.from_delayed(
                    get_tile(level, c, r), shape=tile_shape, dtype=sample_dtype
                )
                row_of_tiles.append(dask_tile)

            # Horizontally concatenate the tiles to form a single row array
            concatenated_row = da.concatenate(row_of_tiles, axis=1)
            rows_of_arrays.append(concatenated_row)

        # Vertically concatenate the rows to form the full level array
        level_dask_array = da.concatenate(rows_of_arrays, axis=0)

        pyramid.append(level_dask_array)

    # --- 4. Define metadata and return the layer data ---
    layer_name = os.path.splitext(os.path.basename(path))[0]
    add_kwargs = {
        "name": layer_name,
        "multiscale": True,
        "contrast_limits": [0, 255],
        "metadata": {
            "path": path,
            "level_count": num_levels,
            "dimensions": slide.dimensions,
        },
    }

    return [(pyramid, add_kwargs, "image")]
