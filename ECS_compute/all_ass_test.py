import numpy
import os

import numpy as np


def load_ass(path):
    ass_list = os.listdir(path)
    for al in ass_list:
        print(al)
        pth = os.path.join(path, al)
        score = np.load(pth)
        print(score.shape)

ass_path = r'../data/ES_ALL'
load_ass(ass_path)