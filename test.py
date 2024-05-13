import tensorflow as tf

from hypervectors import Hypervector

hv = Hypervector(value=10, size=256)

print(hv.tensor)