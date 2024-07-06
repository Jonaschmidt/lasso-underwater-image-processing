from graphics import Graphic
from hypervectors import Hypervector
from bitstreams import Bitstream

import tensorflow as tf
import os
import glob
import random
import pickle
import re
from alive_progress import alive_bar # type: ignore

files = glob.glob(os.path.join("./LSUI_Graphics/input_conv_1", '*'))

for file in files:
    num = ([int(num) for num in re.findall(r'\d+', file)][1])
        
    with open(file, 'rb') as f:
        conv_g = pickle.load(f)

    with open('./LSUI_Graphics/input/%d.pkl' % num, 'rb') as f:
        raw_g = pickle.load(f)

    print(Hypervector.cos_similarity(raw_g.hv, conv_g.hv))