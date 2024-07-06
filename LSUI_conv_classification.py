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

class_0 = files = glob.glob(os.path.join("./LSUI_Graphics/input_conv_1", '*'))
class_1 = files = glob.glob(os.path.join("./LSUI_Graphics/GT_conv_1", '*'))

train_cases_class_0 = class_0[:int(len(class_0) * SPLIT)]
train_cases_class_1 = class_1[:int(len(class_1) * SPLIT)]

test_cases_class_0 = class_0[-1 * int(len(class_0) * (1 - SPLIT)):]
test_cases_class_1 = class_1[-1 * int(len(class_1) * (1 - SPLIT)):]

test_cases = test_cases_class_0 + test_cases_class_1
random.shuffle(test_cases)

### TRAIN CLASS 0
with alive_bar(len(train_cases_class_0)) as bar:
    bar.title = "training class 0..."

    curr_class_hv = Hypervector(value=0, size=HV_LENGTH)
    for case in train_cases_class_0:
        with open(case, 'rb') as f:
            g = pickle.load(f)

        curr_hv = g.hv
        curr_class_hv = Hypervector(tensor=tf.add(curr_class_hv.tensor, curr_hv.tensor))

        bar()

curr_class_hv.polarize()
class_hvs.append(curr_class_hv)

### TRAIN CLASS 1
with alive_bar(len(train_cases_class_1)) as bar:
    bar.title = "training class 1..."

    curr_class_hv = Hypervector(value=0, size=HV_LENGTH)
    for case in train_cases_class_1:
        with open(case, 'rb') as f:
            g = pickle.load(f)

        curr_hv = g.hv
        curr_class_hv = Hypervector(tensor=tf.add(curr_class_hv.tensor, curr_hv.tensor))

        bar()

curr_class_hv.polarize()
class_hvs.append(curr_class_hv)

### TEST
TEST_SAMPLES = len(test_cases)

acc = 0
with alive_bar(TEST_SAMPLES) as bar:
    bar.title = "testing..."

    for file in test_cases:
        with open(file, 'rb') as f:
            g = pickle.load(f)

        test_hv = g.hv

        best_sim = [-1.1, -1]
        for _class in range(NUM_CLASSES):
            cos_sim = Hypervector.cos_similarity(class_hvs[_class], test_hv)

            if(cos_sim > best_sim[0]):
                best_sim[0] = cos_sim
                best_sim[1] = _class

        which_class = 0
        if 'GT' in file:
            which_class = 1

        if(best_sim[1] == which_class):
            acc += 1

        bar()

acc_percent = (acc / TEST_SAMPLES) * 100

print("\nNUM_CLASSES:", NUM_CLASSES)
print("HV_LENGTH:", HV_LENGTH)
print("TEST_SAMPLES", TEST_SAMPLES)

print("\naccuracy: %f" % acc_percent)
print("(%d/%d)" % (acc, TEST_SAMPLES))

print("\ndone")