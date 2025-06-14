"""
This module implements a reader plugin for napari that reads SVS files using openslide.
"""

import numpy as np
import openslide


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str
        Path to file

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # A path list does not make sense for this plugin
        return None

    # otherwise we return the *function* that can read ``path``.
    if path.endswith(".svs"):
        return reader_function

    return None


def reader_function(path):
    """Take a path and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    slide = openslide.OpenSlide(path)
    num_levels = slide.level_count

    myPyramid = []
    # Loop through each level and read the image data
    for level in range(num_levels):
        # Get dimensions for the current level
        level_dimensions = slide.level_dimensions[level]
        print(f"Reading Level {level} with dimensions: {level_dimensions}")

        # Read the entire region for the current level
        region = slide.read_region((0, 0), level, level_dimensions)

        # Convert the region to a numpy array
        region_array = np.array(region)

        # Append the numpy array to the list
        myPyramid.append(region_array)

    add_kwargs = {
        "multiscale": True,
        "contrast_limits": [0, 255],
    }
    layer_type = "image"
    return [(myPyramid, add_kwargs, layer_type)]

if __name__ == "__main__":
    # This is just for testing purposes, you can run this script directly
    # to see if it works.
    path = "TCGA-AK-3440-01Z-00-DX1.ea0763b1-4262-4c75-9f76-080af1ffeab7.svs"
    layers = reader_function(path)
    print("Layers read:", layers)