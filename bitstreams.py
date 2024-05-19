# TODO: docstrings
# TODO: support for binary and polar bitstreams?

import tensorflow as tf
from typing import Optional, Tuple

class Bitstream:
    '''
    A class to represent a bitstream.

    Attributes:
        size (int): The size of the bitstream
        value (int): The value represented by the bitstream
        value_range (Tuple[int, int]): Default (0, size), is used when generating the tensor and is most useful when the bitstream can represent negative numbers
        tensor (tensorflow.python.framework.ops.EagerTensor): The actual bitstream, represented as a tensor
    '''

    def __init__(self, size: int, value: int, value_range: Optional[Tuple[int, int]] = None):
        '''
        Initializes a bitstream with the given size, value, and value range.

        Args:
            size (int): The size of the bitstream, must be positive.
            value (Optional[int]): The value represented by the bitstream. Default is None, which sets the value to half the size.
            value_range (Optional[Tuple[int, int]]): The range of possible values the bitstream can represent. Default is None, which sets the range to (0, size).
        '''
        if size <= 0:
            raise ValueError("Size must be a positive integer.")

        self.size = size
        self.value = value
        self.tensor = self._generate_tensor(value_range)

    def _generate_tensor(self, value_range: Optional[Tuple[int, int]] = None) -> tf.Tensor:
        '''
        Generates the tensor representation of the bitstream.

        Args:
            value_range (Optional[Tuple[int, int]]): The range of possible values the bitstream can represent. Default is None, which sets the range to (0, size).

        Returns:
            tf.Tensor: The generated tensor representing the bitstream.
        '''
        if value_range is None:
            value_range = (0, self.size)

        min_value, max_value = value_range
        max_value += 1

        num_ones = int(self.size * (self.value + min_value) / (max_value - 1))
        corr_vec = tf.constant([-1] * self.size)
        indices = tf.random.shuffle(tf.range(self.size))[:num_ones]
        updates = tf.ones(num_ones, dtype=tf.int32)

        return tf.tensor_scatter_nd_update(corr_vec, tf.expand_dims(indices, 1), updates)

    # TODO: def regen_tensor()

    def binarize(self):
        '''
        Converts the tensor to a binary representation where all positive values become 1 and non-positive values become 0.
        '''
        self.tensor = tf.where(self.tensor > 0, tf.ones_like(self.tensor), tf.zeros_like(self.tensor))

    def polarize(self):
        '''
        Converts the tensor to a polar representation where all non-positive values become -1 and positive values become 1.
        '''
        self.tensor = tf.where(self.tensor <= 0, tf.constant(-1, dtype=tf.int32), tf.ones_like(self.tensor))

    # returns [-1, 1], with 0 denoting complete orthogonality
    @staticmethod
    def cos_similarity(bs1, bs2) -> float:
        '''
        Computes the cosine similarity between two bitstreams.

        Args:
            bs1 (Bitstream): The first bitstream.
            bs2 (Bitstream): The second bitstream.

        Returns:
            float: The cosine similarity between hv1 and hv2, ranging from -1 to 1, where 0 denotes complete orthogonality.
        '''
        return (tf.keras.losses.cosine_similarity(tf.cast(hv1.tensor, dtype=tf.float32), tf.cast(hv2.tensor, dtype=tf.float32))).numpy()    
