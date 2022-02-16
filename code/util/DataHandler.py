# -*- coding: utf-8 -*-

import random

import numpy as np

from entity.Cell import Cell


class DataHandler:
    def __init__(self):
        self.misRowIndexList = []

    def genMisCel(self, db, label_idx):
        db = db.drop(label_idx, axis=1).values
        size, attr_size = db.shape
        mask = np.ones((size, attr_size), dtype=np.int8)
        cells = []
        for r in range(size):
            for c in range(attr_size):
                if db[r, c] != db[r, c]:
                    pos = (r, c)
                    mask[r, c] = 0
                    cell = Cell(pos)
                    cells.append(cell)
        cells = sorted(cells, key=lambda x: (x.position[0], x.position[1]))
        return cells, mask

    def genMisCelMul(self, db, label_idx, misNum, size, seed, selList):
        db = db.drop(label_idx, axis=1).values
        size, attr_size = db.shape

        mask = np.ones((size, attr_size), dtype=np.int8)
        random.seed(seed)
        self.misRowIndexList = random.sample(range(0, size), misNum)
        cells = []
        for r in self.misRowIndexList:
            for c in selList:
                pos = (r, c)
                mask[r, c] = 0
                truth_value = db[pos]
                cell = Cell(pos)
                cell.setTruth(truth_value)
                cells.append(cell)
        cells = sorted(cells, key=lambda x: (x.position[0], x.position[1]))
        return cells, mask

    def genMissSelMulGivenMisRowList(self):
        pass

    def genMissSelMul(self):
        pass

    def genDelCompRowIndexList(self, cleanRatio, size, seed, errorTupleNum):
        delComSize = int((size - errorTupleNum) * (1 - cleanRatio))
        random.seed(seed)
        delCompRowIndexList = random.sample(set(range(0, size)) - set(self.misRowIndexList), delComSize)
        return delCompRowIndexList

    def getMisCells(self):
        pass

    def getMisRowIndexList(self):
        pass
