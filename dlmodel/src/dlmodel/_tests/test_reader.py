import pytest
import numpy as np
from dask.array import Array
from dlmodel import reader_funtion  # <-- replace with actual import path

@pytest.fixture
def svs_sample_path(tmp_path):
    # This is a placeholder. Use a real .svs path for actual tests.
    return "/archive/bioinformatics/Zhou_lab/shared/rxwu/course/Module_5_group1/data/TCGA-AK-3440-01Z-00-DX1.ea0763b1-4262-4c75-9f76-080af1ffeab7.svs"  # Replace with your actual test file

def test_reader_function_valid_layers(svs_sample_path):
    reader = reader_funtion(svs_sample_path)
    assert callable(reader), "Reader function not returned for valid SVS path"

    layers = reader(svs_sample_path)
    assert isinstance(layers, list), "Returned object is not a list"

    # Expected structure: [(pyramid_rgb, kwargs_img, 'image'), (label_mask, kwargs_mask, 'image')]
    assert len(layers) == 2, "Expected 2 layers: RGB + StarDist mask"

    rgb_layer, mask_layer = layers

    # Check RGB pyramid
    data_rgb, meta_rgb, type_rgb = rgb_layer
    assert type_rgb == "image"
    assert isinstance(data_rgb, list)
    assert all(isinstance(lvl, Array) for lvl in data_rgb), "RGB pyramid is not dask arrays"
    assert meta_rgb.get("multiscale", False), "RGB image not marked as multiscale"

    # Check StarDist mask layer
    data_mask, meta_mask, type_mask = mask_layer
    assert type_mask == "image"
    assert isinstance(data_mask, Array), "StarDist mask is not a dask array"
    assert "StarDist" in meta_mask["name"]

    # Check contrast limits are set
    assert "contrast_limits" in meta_rgb and isinstance(meta_rgb["contrast_limits"], list)
    assert "contrast_limits" in meta_mask and isinstance(meta_mask["contrast_limits"], list)
