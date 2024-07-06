from graphics import Graphic
from hypervectors import Hypervector
from bitstreams import Bitstream

import tensorflow as tf
import os
import glob
import random
import pickle
from alive_progress import alive_bar # type: ignore

NUM_CLASSES = 2
HV_LENGTH = 10_000
SPLIT = 0.80
class_hvs = []

class_0 = files = glob.glob(os.path.join("./LSUI_Graphics/input", '*'))
class_1 = files = glob.glob(os.path.join("./LSUI_Graphics/GT", '*'))

train_cases_class_0 = class_0[:int(len(class_0) * SPLIT)]
train_cases_class_1 = class_1[:int(len(class_1) * SPLIT)]

test_cases_class_0 = class_0[-1 * int(len(class_0) * (1 - SPLIT)):]
test_cases_class_1 = class_1[-1 * int(len(class_1) * (1 - SPLIT)):]

test_cases = test_cases_class_0 + test_cases_class_1
random.shuffle(test_cases)

print(test_cases)