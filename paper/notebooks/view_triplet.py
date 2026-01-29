import napari
import mrcfile
import numpy as np
clone, degraded, perfect = mrcfile.read('triplet_clone.mrc'), mrcfile.read('triplet_degraded.mrc'), mrcfile.read('triplet_perfect.mrc')

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(perfect, contrast_limits=[-4,4], gamma=.5, rendering='minip')
viewer.add_image(degraded, contrast_limits=[-4,4], gamma=.5, rendering='minip')
viewer.add_image(np.flip(clone), contrast_limits=[-4,4], gamma=.5, rendering='minip')

viewer.camera.angles = (21, -44, -29)
napari.run()
