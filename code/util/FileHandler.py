# -*- coding: utf-8 -*-


import pandas as pd
import os
import numpy as np


class FileHandler:
    def __init__(self):
        self.PATH = 'data/'

    def readCompData(self, input, header=None, index_col=None):
        data = pd.read_csv(os.path.join(self.PATH, input), dtype=str, header=header, index_col=index_col)
        return pd.DataFrame(np.array(data))
