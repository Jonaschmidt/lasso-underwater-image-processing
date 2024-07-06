from graphics import Graphic
from hypervectors import Hypervector
from bitstreams import Bitstream

import tensorflow as tf
import os
import glob
import random
from alive_progress import alive_bar # type: ignore
import pickle
import re

BATCH_SIZE = 8
HV_LENGTH = 10_000
RESIZE_DIMS = (64, 64)

'''
raws = glob.glob(os.path.join("./LSUI/input", '*'))
raws_split = [raws[i:i + int(BATCH_SIZE/2)] for i in range(0, len(raws), int(BATCH_SIZE/2))]

with alive_bar(len(raws_split)) as bar:
    for sub_list in raws_split:

        for raw in sub_list:
            idx = [int(num) for num in re.findall(r'\d+', raw)][0]

            proc = raw.replace("input", "GT")

            try:
                raw_g = Graphic(file_path=raw, hv_size=HV_LENGTH, bar_show=False, resize_dims=RESIZE_DIMS)
                with open('./LSUI_Graphics/input/%d.pkl' % idx, 'wb') as f:
                    pickle.dump(raw_g, f)

                proc_g = Graphic(file_path=proc, hv_size=HV_LENGTH, bar_show=False, resize_dims=RESIZE_DIMS)
                with open('./LSUI_Graphics/GT/%d.pkl' % idx, 'wb') as f:
                    pickle.dump(raw_g, f)

            finally:
                pass

        bar()
'''

raws = glob.glob(os.path.join("./LSUI_Graphics/input", '*'))
raws_split = [raws[i:i + int(BATCH_SIZE/2)] for i in range(0, len(raws), int(BATCH_SIZE/2))]

with alive_bar(len(raws_split)) as bar:
    for sub_list in raws_split:

        for raw in sub_list:
            idx = [int(num) for num in re.findall(r'\d+', raw)][0]

            proc = raw.replace("input", "GT")

            with open(raw, 'rb') as f:
                p_raw = pickle.load(f)
            p_raw.convolute_bs_array()

            with open(proc, 'rb') as f:
                p_proc = pickle.load(f)
            p_proc.convolute_bs_array()

            try:
                with open('./LSUI_Graphics/input_conv_1/%d.pkl' % idx, 'wb') as f:
                    pickle.dump(p_raw, f)

                with open('./LSUI_Graphics/GT_conv_1/%d.pkl' % idx, 'wb') as f:
                    pickle.dump(p_proc, f)

            finally:
                pass

        bar()

