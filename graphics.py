# TODO: docstrings

from PIL import Image
from alive_progress import alive_bar
from typing import Optional
import numpy as np
import tensorflow as tf

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

    @staticmethod
    def _mux_8bit(arr, selector_arr):
        selector_to_bin = ''.join(selector_arr.astype(str))
        selector_to_dec = int(selector_to_bin, 2)
        return arr[selector_to_dec]

    @staticmethod
    def _mux_2bit(arr, selector):
        return arr[selector]

    # TODO
    # TODO: prevent convolution is impossible
    # 3x3 convolution
    def convolute_bs_array(self):
        selector_depth = self.bs_array[0, 0].tensor.shape[0]
        
        val = selector_depth / 2
        selector0 = Bitstream(size=selector_depth, value=val)
        selector1 = Bitstream(size=selector_depth, value=val)
        selector2 = Bitstream(size=selector_depth, value=val)

        val = (8 * selector_depth) / 9
        selector3 = Bitstream(size=selector_depth, value=val)

        selector0.binarize()
        selector1.binarize()
        selector2.binarize()
        selector3.binarize()

        new_shape = self.bs_array.shape - np.array([2, 2])
        new_bs_arr = np.empty(new_shape, dtype=object)

        width_passes = self.bs_array.shape[0] - 2
        height_passes = self.bs_array.shape[1] - 2

        #bs_tensor = tf.convert_to_tensor(self.bs_array)

        with alive_bar(width_passes * height_passes * selector_depth) as bar:
            bar.title = "convoluting..."

            for w in range(width_passes):
                for h in range(height_passes):

                    depth_window = self.bs_array[w:(w + 3), h:(h + 3)].flatten()

                    to_be_tensor = []
                    for d in range(selector_depth):
                        window = [
                            depth_window[0].tensor[d].numpy(),
                            depth_window[1].tensor[d].numpy(),
                            depth_window[2].tensor[d].numpy(),
                            depth_window[3].tensor[d].numpy(),
                            depth_window[4].tensor[d].numpy(),
                            depth_window[5].tensor[d].numpy(),
                            depth_window[6].tensor[d].numpy(),
                            depth_window[7].tensor[d].numpy(),
                            depth_window[8].tensor[d].numpy()
                        ]
                        
                        arr = window[:8]
                        selector_arr = np.array([selector0.tensor[d], selector1.tensor[d], selector2.tensor[d]])

                        bit = self._mux_8bit(arr, selector_arr)

                        arr = [bit, window[8]]
                        selector = selector3.tensor[d]

                        bit = self._mux_2bit(arr, selector)

                        to_be_tensor.append(bit)
                        
                        bar()

                    tensor = tf.convert_to_tensor(to_be_tensor, dtype=tf.int32)
                    bs = Bitstream(tensor=tensor)

                    new_bs_arr[w, h] = bs

        values = np.array([[obj.value for obj in row] for row in new_bs_arr], dtype=np.uint8)
        image = Image.fromarray(values, mode='L')

        self.bs_array = new_bs_arr
        self.image = image

