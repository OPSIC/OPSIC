from time import time

import numpy as np
import os
import pandas as pd
import warnings

from algorithm.ERACER import ERACERNum
from algorithm.GMM import GMM
from algorithm.kNNE import KNNE
from algorithm.CMI import CMI
from algorithm.MICE import MICE
from algorithm.BaseMissing import BaseMissing
from algorithm.OPSICAppro import OPSICAppro
from algorithm.OPSICExact import OPSICExact
from algorithm.IFC import IFC
from algorithm.CI import ClustImpute
from algorithm.NMF import NMF
from util.Assist import Assist
from util.DataHandler import DataHandler as dh
from util.FileHandler import FileHandler as fh

warnings.filterwarnings("ignore")


class CompTest:
    def __init__(self, filename, label_idx=-1, exp_cluster=None, header=None, index_col=None):
        self.name = filename.split('/')[-1].split('.')[0]
        self.REPEATLEN = 1
        self.RBEGIN = 998

        self.misTupleNum = 30
        self.selCans = [6]
        self.exp_cluster = exp_cluster
        self.label_idx = label_idx
        self.db = fh().readCompData(filename, header=header, index_col=index_col)
        self.dh = dh()
        self.totalsize = self.db.shape[0]
        self.size = self.totalsize * 1

        self.misAttrNum = len(self.selCans)
        self.selList = [self.selCans[mi] for mi in range(self.misAttrNum)]

        self.ratioList = np.arange(1, 1.02, 0.1)

        self.RATIOSIZE = len(self.ratioList)

        self.ALGNUM = 11

        self.alg_flags = [True, True, True, True, True, True, True, True, True, True, True]

        self.totalTime = np.zeros((self.RATIOSIZE, self.ALGNUM))
        self.totalCost = np.zeros((self.RATIOSIZE, self.ALGNUM))
        self.parasAlg = np.zeros((self.ALGNUM))
        self.purityAlg = np.zeros((self.ALGNUM))
        self.fmeasureAlg = np.zeros((self.ALGNUM))
        self.NMIAlg = np.zeros((self.ALGNUM))
        self.totalpurity = np.zeros((self.RATIOSIZE, self.ALGNUM))
        self.totalfmeasure = np.zeros((self.RATIOSIZE, self.ALGNUM))
        self.totalNMI = np.zeros((self.RATIOSIZE, self.ALGNUM))

        self.features = [[0], [1]]

        self.K = 30
        self.N_Cluster = 10

        self.ERACER_K = 10
        self.ERACER_maxIter = 500
        self.ERACER_threshold = 0.1

        self.K_Candidate = 20
        self.L = 20
        self.C = 1

        self.IFC_min_k = 3
        self.IFC_max_k = 10
        self.IFC_maxIter = 20
        self.IFC_threshold = 0.01

        self.CI_K = 7
        self.CI_maxIter = 20
        self.CI_n_end = 10
        self.CI_c_steps = 10

        self.K_OPSIC = 3
        self.K_Candidate_OPSIC = 2
        self.epsilon_OPSIC = 0.2

    def Dirty_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("Dirty begin!")
        algIndex = 0
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)
                dirty = BaseMissing(self.db, self.label_idx, self.exp_cluster, cells, mask)

                dirty.setDelCompRowIndexList(delCompRowIndexList)
                dirty.initVals()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = dirty.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("Dirty over!")

    def KNNE_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("KNNE begin!")
        algIndex = 1
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                knne = KNNE(self.db, self.label_idx, self.exp_cluster, cells, mask, self.K, self.features)
                knne.setDelCompRowIndexList(delCompRowIndexList)
                knne.mainKNNE()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += knne.getAlgtime()

                origin_y, modify_y = knne.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("KNNE over!")

    def GMM_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("GMM begin!")
        algIndex = 2
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                gmm = GMM(self.db, self.label_idx, self.exp_cluster, cells, mask, self.K, self.N_Cluster, seed)

                gmm.setDelCompRowIndexList(delCompRowIndexList)
                gmm.mainGMM()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += gmm.getAlgtime()

                origin_y, modify_y = gmm.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("GMM over!")

    def ERACER_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("ERACER begin!")
        algIndex = 3
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                eracer = ERACERNum(self.db, self.label_idx, self.exp_cluster, cells, mask)
                eracer.setParams(self.ERACER_K, self.ERACER_maxIter, self.ERACER_threshold)
                eracer.setDelCompRowIndexList(delCompRowIndexList)
                eracer.mainEracer()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += eracer.getAlgtime()

                origin_y, modify_y = eracer.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("ERACER over!")

    def IFC_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("IFC begin!")
        algIndex = 4
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                ifc = IFC(self.db, self.label_idx, self.exp_cluster, cells, mask)
                ifc.setDelCompRowIndexList(delCompRowIndexList)
                ifc.setParams(self.IFC_min_k, self.IFC_max_k, self.IFC_maxIter, self.IFC_threshold)
                ifc.mainIFC()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += ifc.getAlgtime()

                origin_y, modify_y = ifc.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("IFC over!")

    def ClustImpute_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("ClustImpute begin!")
        algIndex = 5
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                CI = ClustImpute(self.db, self.label_idx, self.exp_cluster, cells, mask)
                CI.setDelCompRowIndexList(delCompRowIndexList)
                CI.setParams(self.CI_K, self.CI_maxIter, self.CI_n_end, self.CI_c_steps)
                CI.mainClustImpute()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += CI.getAlgtime()

                origin_y, modify_y = CI.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("ClustImpute over!")

    def CMI_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("CMI begin!")
        algIndex = 6
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                cmi = CMI(self.db, self.label_idx, self.exp_cluster, cells, mask)
                cmi.setCenterRatio(0.02)
                cmi.setSeed(seed)

                cmi.setDelCompRowIndexList(delCompRowIndexList)
                cmi.mainCMI()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += cmi.getAlgtime()

                origin_y, modify_y = cmi.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("CMI over!")

    def MICE_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("MICE begin!")
        algIndex = 7
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                mice = MICE(self.db, self.label_idx, self.exp_cluster, cells, mask)

                mice.setDelCompRowIndexList(delCompRowIndexList)
                mice.main_mice()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += mice.getAlgtime()

                origin_y, modify_y = mice.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("MICE over!")

    def NMF_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("SparseNMF begin!")
        algIndex = 8
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                nmf = NMF(self.db, self.label_idx, self.exp_cluster, cells, mask, self.N_Cluster, seed)
                nmf.setDelCompRowIndexList(delCompRowIndexList)
                y, origin_y = nmf.mainSNMF()
                modify_y = np.argmax(y, axis=1)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("SNMF over!")


    def OPSIC_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("OPSIC begin!")
        algIndex = 9
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)
                opsic = OPSICAppro(self.db, self.label_idx, self.exp_cluster, cells, mask)
                opsic.setK(self.K_OPSIC)
                opsic.setK_Candidate(self.K_Candidate_OPSIC)
                opsic.setEpsilon(self.epsilon_OPSIC)
                opsic.setDelCompRowIndexList(delCompRowIndexList)
                opsic.mainOPSIC()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += opsic.getAlgtime()

                cluster_dict = {}
                for i in range(len(opsic.clusterMembersList)):
                    for j in opsic.clusterMembersList[i]:
                        cluster_dict[j] = i
                modify_y = []
                for j in opsic.compRowIndexList + opsic.misRowIndexList:
                    modify_y.append(cluster_dict[j])
                print(modify_y)
                origin_y, modify_y_1 = opsic.modify_down_stream(cells)
                print(list(origin_y))
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("OPSIC over")

    def OPSIC_exact_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("OPSIC_exact begin!")
        algIndex = 10
        if self.alg_flags[algIndex]:
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)
                opsic = OPSICExact(self.db, self.label_idx, self.exp_cluster, cells, mask)

                opsic.setK(self.K_OPSIC)
                opsic.setK_Candidate(self.K_Candidate_OPSIC)
                opsic.setEpsilon(self.epsilon_OPSIC)
                opsic.setDelCompRowIndexList(delCompRowIndexList)
                opsic.mainOPSIC()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += opsic.getAlgtime()

                cluster_dict = {}
                for i in range(len(opsic.clusterMembersList)):
                    for j in opsic.clusterMembersList[i]:
                        cluster_dict[j] = i
                modify_y = []
                for j in opsic.compRowIndexList + opsic.misRowIndexList:
                    modify_y.append(cluster_dict[j])
                print(modify_y)
                origin_y, modify_y_1 = opsic.modify_down_stream(cells)
                print(list(origin_y))
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]

        print("OPSIC_exact over")

    def alg_exp(self):
        for ratioIndex in range(self.RATIOSIZE):
            compRatio = self.ratioList[ratioIndex]

            print("size:" + str(self.size) + ", compRatio:" + str(compRatio) + " begin...")
            self.Dirty_exp(ratioIndex)
            self.KNNE_exp(ratioIndex)
            self.GMM_exp(ratioIndex)
            self.ERACER_exp(ratioIndex)
            self.IFC_exp(ratioIndex)
            self.ClustImpute_exp(ratioIndex)
            self.CMI_exp(ratioIndex)
            self.MICE_exp(ratioIndex)
            self.NMF_exp(ratioIndex)
            self.OPSIC_exp(ratioIndex)
            self.OPSIC_exact_exp(ratioIndex)

            name1 = self.name + '_K' + str(self.K_OPSIC) + '_Candidate' + str(self.K_Candidate_OPSIC) + '_epsilon' + str(self.epsilon_OPSIC) + '_selCans' + str(self.selCans)
            name2 = self.name
            ratio_arr = np.array(self.ratioList).reshape(-1, 1)
            cost_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalCost / self.REPEATLEN), axis=1),
                                   columns=["CompRatio", "Dirty", "kNNE", "GMM", "ERACER", "IFC", "CI", "CMI", "MICE", "NMF", "OPSIC", "OPSIC_exact"])
            time_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalTime / self.REPEATLEN), axis=1),
                                   columns=["CompRatio", "Dirty", "kNNE", "GMM", "ERACER", "IFC", "CI", "CMI", "MICE", "NMF", "OPSIC", "OPSIC_exact"])

            purity_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalpurity / self.REPEATLEN), axis=1),
                                     columns=["CompRatio", "Dirty", "kNNE", "GMM", "ERACER", "IFC", "CI", "CMI", "MICE", "NMF", "OPSIC", "OPSIC_exact"])

            if not os.path.exists(os.path.join("result/compare", name1)):
                os.makedirs(os.path.join("result/compare", name1))

            cost_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_cost" + ".tsv", sep='\t',
                           float_format="%.3f",
                           index=False)
            time_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_time" + ".tsv", sep='\t',
                           float_format="%.3f",
                           index=False)
            purity_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_purity" + ".tsv", sep='\t',
                             float_format="%.3f",
                             index=False)

        print("all over!")


if __name__ == '__main__':
    WineCompTest = CompTest("../data/Wine/wine.data", label_idx=0, exp_cluster=3, header=None, index_col=None)
    WineCompTest.alg_exp()
