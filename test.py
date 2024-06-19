import numpy as np
import tensorflow as tf
from PIL import Image

from graphics import Graphic
from bitstreams import Bitstream

g = Graphic("./test_graphics/10008.png")

print(g.bs_array)


tensor_list = []

width, height = g.bs_array.shape
for w in range(width):
    for h in range(height):
        tensor_list.append(g.bs_array[w, h].tensor)

depth = tensor_list[0].shape
tensor = tf.stack(tensor_list)
tensor = tf.reshape(tensor, (width, height, *depth))

print()
print(tensor[0, 0, :].shape)

print()
print(tensor[0, 0, :] == g.bs_array[0, 0].tensor)

'''
values = np.array([[obj.value for obj in row] for row in g.bs_array], dtype=np.uint8)
image = Image.fromarray(values, mode='L')

image.show(title="unconvoluted")
'''

g.show()
g.convolute_bs_array()
g.show()
g.convolute_bs_array()
g.show()
g.convolute_bs_array()
g.show()
g.convolute_bs_array()
g.show()
g.convolute_bs_array()
g.show()

