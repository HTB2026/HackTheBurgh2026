"""tetra3 (cedar-solve 0.5.1), vendored: an offline "lost in space" plate solver.

From https://github.com/smroid/cedar-solve (a fork of https://github.com/esa/tetra3),
Apache License 2.0 (LICENSE.txt). Vendored rather than installed because the
package pins numpy<2 and Pillow<9. Local changes: ``np.math.factorial`` ->
``math.factorial`` (removed in numpy 2), ``scipy.ndimage.filters.*`` ->
``scipy.ndimage.*`` (deprecated namespace) and relative imports. The bundled
database (data/default_database.npz) covers fields of view of 10-30 degrees.
"""

from .tetra3 import Tetra3
from .tetra3 import crop_and_downsample_image
from .tetra3 import get_centroids_from_image

__all__ = ["Tetra3", "get_centroids_from_image", "crop_and_downsample_image"]
