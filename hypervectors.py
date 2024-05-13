import tensorflow as tf
from typing import Optional, Tuple

class Hypervector:
    def __init__(self, value: int, size: int, value_range: Optional[Tuple[int, int]] = None):
        self.value = value
        self.size = size
        self.tensor = self.generate_tensor(value_range)

    def generate_tensor(self, value_range: Optional[Tuple[int, int]] = None) -> tf.Tensor:
        if value_range is None:
            value_range = (0, self.size)

        min_value, max_value = value_range
        max_value += 1

        num_ones = int(self.size * (self.value + min_value) / (max_value - 1))
        corr_vec = tf.constant([-1] * self.size)
        indices = tf.random.shuffle(tf.range(self.size))[:num_ones]
        updates = tf.ones(num_ones, dtype=tf.int32)

        return tf.tensor_scatter_nd_update(corr_vec, tf.expand_dims(indices, 1), updates)
    
    def regen_tensor(self) -> None:
        # Shuffle the indices again to ensure randomness
        indices = tf.random.shuffle(tf.range(self.size))[:tf.reduce_sum(self.tensor == 1)]
        updates = tf.ones(tf.shape(indices), dtype=tf.int32)
        self.tensor = tf.tensor_scatter_nd_update(self.tensor, tf.expand_dims(indices, 1), updates)


