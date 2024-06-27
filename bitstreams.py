# TODO: docstrings
# TODO: support for binary and polar bitstreams?

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple

# TODO: docstrings
class Bitstream:
    # TODO: update these docstrings
    '''
    A class to represent a bitstream.

    Attributes:
        size (int): The size of the bitstream
        value (int): The value represented by the bitstream
        tensor (tensorflow.python.framework.ops.EagerTensor): The actual bitstream, represented as a tensor
    '''

    def __init__(self, size: Optional[int]=256, value: Optional[int]=None, tensor:Optional[tf.Tensor]=None):
        '''
        Initializes a bitstream with the given size, value, and value range.

        Args:
            size (int): The size of the bitstream, must be positive. Will be ignored if tensor is passed instead of value. Default is 256 if value is passed.
            value (Optional[int]): The value represented by the bitstream. Default is None, which will generate value based on given tensor attribute. Only one of tensor and value must be provided.
                *It is generally recommended to pass value and size intstead of tensor*
            tensor (Optional[tf.Tensor]): A Tensor representing the actual bitstream. Default is None, which will generate tensor based on given value. Only one of tensor and value must be provided.
        '''

        if size <= 0:
            raise ValueError("Size must be a positive integer.")
        
        self.size = size

        if value == None and tensor == None:
            value = 0

        if value != None and tensor != None:
            raise ValueError("At least one or none of 'tensor' or 'value' must be provided.")
        
        if value != None:
            self.value = value
            self.tensor = self._generate_tensor()
        else:
            self.tensor = tensor
            self.value = self._generate_value()
            self.size = self.tensor.shape[0]


    def _generate_tensor(self) -> tf.Tensor:
        '''
        Generates the tensor representation of the bitstream.

        Returns:
            tf.Tensor: The generated tensor representing the bitstream.
        '''
        value_range = (0, self.size)

        min_value, max_value = value_range
        max_value += 1

        num_ones = int(self.size * (self.value + min_value) / (max_value - 1))
        corr_vec = tf.constant([-1] * self.size)
        indices = tf.random.shuffle(tf.range(self.size))[:num_ones]
        updates = tf.ones(num_ones, dtype=tf.int32)

        return tf.tensor_scatter_nd_update(corr_vec, tf.expand_dims(indices, 1), updates)

    def _generate_value(self) -> int:
        '''
        Generates the value of the a passed bitstream (Tensor).

        Returns:
            int: The generated value based on the number of 1's in the Tensor passed.
        '''
        return tf.reduce_sum(tf.cast(tf.equal(self.tensor, 1), tf.int32)).numpy()

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
            float: The cosine similarity between bs1 and bs2, ranging from -1 to 1, where 0 denotes complete orthogonality.
        '''
        return (tf.keras.losses.cosine_similarity(tf.cast(bs1.tensor, dtype=tf.float32), tf.cast(bs2.tensor, dtype=tf.float32))).numpy()    

