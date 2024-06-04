# TODO: docstrings

from PIL import Image
from typing import Optional
import numpy as np

from hypervectors import Hypervector
from bitstreams import Bitstream

# filepath
# image
# label
# description (optional)
# array
# bs_array
# hv

# show()

class Graphic:
    def __init__(self, file_path: str, label: Optional[str]=None, description: Optional[str]=None):
        self.file_path = file_path
        try:
            self.image = Image.open(file_path)  # load the image using PIL
        except (IOError, OSError) as e:
            raise FileNotFoundError(f"Error opening image at {file_path}: {e}")
        
        self.label = label

        self.description = description

        # note that arrays are indexed differently than PIL Images with the getpixel() function
        self.array = np.asarray(self.image)

        self.bs_array = self._gen_bs_array()

        self.hv = self._gen_hypervector()

    def _gen_bs_array(self):
        bs_array = np.empty(self.array.shape, dtype=Bitstream)

        for idx in np.ndindex(bs_array.shape):
            bs_array[idx] = Bitstream(size=256, value=self.array[idx])

        return bs_array

    def _gen_hypervector(self):
        return None

    def show(self):
        self.image.show(title="Graphic")

