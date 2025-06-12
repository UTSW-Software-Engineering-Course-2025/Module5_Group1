import matplotlib.pyplot as plt
import pytest

from svsreader import napari_get_reader

TEST_SVS_PATH = r"D:/utsw-study\softwareE\module5/napari-hello\TCGA-AK-3440-01Z-00-DX1.ea0763b1-4262-4c75-9f76-080af1ffeab7.svs"


@pytest.mark.skipif(
    True, reason="Enable this only when a real SVS file is available"
)
def test_reader_real_file():
    """Test reader using a real SVS file."""
    reader = napari_get_reader(TEST_SVS_PATH)
    assert callable(reader)

    layer_data_list = reader(TEST_SVS_PATH)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0

    data, meta, layer_type = layer_data_list[0]
    assert data.ndim == 3  # Expect RGB
    assert layer_type == "image"
    assert "level_count" in meta
    assert "dimensions" in meta


def test_get_reader_pass():
    """Test that non-SVS files are correctly rejected."""
    reader = napari_get_reader("fake.file")
    assert reader is None


def test_fake_svs_extension(tmp_path):
    """Test that .svs extension triggers reader, even with fake data (for structural test)."""
    fake_svs_file = tmp_path / "fake_slide.svs"
    fake_svs_file.write_text("this is not a real svs file")

    reader = napari_get_reader(str(fake_svs_file))
    assert callable(reader)

    # 调试模式下你可以在 reader_function 中跳过 OpenSlide 部分并返回假的 LayerData
    try:
        layer_data_list = reader(str(fake_svs_file))
    except Exception:
        # 如果你没有 stub/mock OpenSlide，可以允许抛异常
        pytest.skip("OpenSlide cannot open fake file")
        return

    assert isinstance(layer_data_list, list)


# debug_reader.py


test_path = r"D:/utsw-study\softwareE\module5/napari-hello\TCGA-AK-3440-01Z-00-DX1.ea0763b1-4262-4c75-9f76-080af1ffeab7.svs"

reader = napari_get_reader(test_path)
if reader is None:
    print("Reader not found or unsupported file.")
else:
    layers = reader(test_path)
    for data, meta, layer_type in layers:
        print(f"Layer type: {layer_type}")
        print(f"Metadata: {meta}")
        print(
            f"Data shape: {data[0].shape}"
        )  # Access the first element of the data list

        # visualize the first layer data
        plt.imshow(data[0])  # Access the first element of the data list
        plt.title(meta.get("name", "SVS Image"))
        plt.show()
