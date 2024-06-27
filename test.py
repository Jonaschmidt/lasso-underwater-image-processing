import tensorflow as tf

from hypervectors import Hypervector

tensor1 = tf.constant([1.0, 1.0, 1.0])
tensor2 = tf.constant([1.0, 1.0, 1.0])

dot_product = tf.reduce_sum(tensor1 * tensor2)

norm_tensor1 = tf.norm(tensor1)
norm_tensor2 = tf.norm(tensor2)

cos_sim = dot_product / (norm_tensor1 * norm_tensor2)

print("Cosine Similarity:", cos_sim.numpy())

foo = Hypervector(tensor=tensor1)
bar = Hypervector(tensor=tensor2)

print(Hypervector.cos_similarity(foo, bar))