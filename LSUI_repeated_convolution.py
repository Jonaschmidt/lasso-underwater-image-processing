import glob
import os
import re
import pickle

raws = glob.glob(os.path.join("./LSUI_Graphics/input_conv_1", '*'))
convs = glob.glob(os.path.join("./LSUI_Graphics/input_conv_2", '*'))

for idx in range(len(raws)):

    num = [int(num) for num in re.findall(r'\d+', raws[idx])][1]

    if not "./LSUI_Graphics/input_conv_2/%d.pkl" % num in convs:

        print("convoluting", num)

        with open("./LSUI_Graphics/input_conv_1/%d.pkl" % num, 'rb') as f:
            g = pickle.load(f)
            g.bar_show = True
        
        try:
            g.convolute_bs_array()
            g.bar_show = False

            with open("./LSUI_Graphics/input_conv_2/%d.pkl" % num, 'wb') as f:
                pickle.dump(g, f)

            with open("./LSUI_Graphics/GT_conv_1/%d.pkl" % num, 'rb') as f:
                g = pickle.load(f)
                g.bar_show = True

            g.convolute_bs_array()
            g.bar_show = False

            with open("./LSUI_Graphics/GT_conv_2/%d.pkl" % num, 'wb') as f:
                pickle.dump(g, f)

        finally:
            break

print(num, "done")

