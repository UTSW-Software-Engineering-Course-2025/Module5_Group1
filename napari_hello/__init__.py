from napari.utils.notifications import show_info
import napari as npr
viewer=npr.Viewer()
viewer.add_image(myPyramid,contrast_limits=[0,255])
def show_hello_message():
    show_info('Hello, world!')
