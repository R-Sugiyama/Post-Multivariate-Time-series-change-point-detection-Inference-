# coding: utf-8
"""
detecting change point, dimension using scan statistic and DP
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy

from . import common_function

#----------------------------------------------------------------------
#変化点tを探すクラス
class Scan():
    def __init__(self, X, M):
        """
        instance variable

        X: データ
        M: 分割セグメント数
        N: 系列の長さ
        d: 系列数
        c, D: 一時的な値を格納するリスト
        B: 部分問題における変化点を格納する行列
        dimansion: 部分問題における変化点の系列集合を格納する多重リスト
        statistic: 部分問題における統計量の総和を格納する行列
        """
        self.X = X #データ
        self.M = M
        self.N = np.shape(X)[1]
        self.d = np.shape(X)[0]
        self.c = [] #記録するだけのリスト
        self.D = [] #記録するだけのリスト
        self.index = []
        self.SortIndex = [[[] for i in range(self.N+1)] for j in range(self.M+1)]
        self.B = np.zeros((self.M+1, self.N+1), dtype=int)
        self.dimension = [[[] for i in range(self.N+1)] for j in range(self.M+1)]
        self.statistic = np.zeros((self.M+1, self.N+1))
        self.detect_change_points()

    def Z_t(self, t, start, end):
        Z = np.sum(self.X[:,start-1:t], axis=1)/(t-start+1) - np.sum(self.X[:,t:end], axis=1)/(end-t)
        Z = np.sqrt((t-start+1)*(end-t)/(end-start+1))*Z
        return Z

    #L_scanを計算する関数(変化点tの中で最もscan統計量をソートして上位がp個の組み合わせを選ぶ)
    def L_scan_sort(self, t, start, end, pre_sum):
        onevec = common_function.OneVec(self.d)
        comb = common_function.Combination(self.d, 1)
        points = comb.combinationList()
        Z = self.Z_t(t, start, end)
        l = np.array([np.dot(onevec.get_2(point)*Z, onevec.get_2(point)*Z) for point in points])
        sort_index = np.argsort(l)[::-1]
        que_index = []
        que_statistic = []
        sum_statistic = 0
        for index in sort_index:
            que_index.append(index)
            self.c.append(copy.deepcopy(que_index))
            sum_statistic += l[index]
            p = len(que_index)
            que_statistic.append((sum_statistic - p)/np.sqrt(2*p) + pre_sum)
        return np.array(que_statistic)

    #各点tに対して一番大きい統計量の組み合わせをとってくる
    def max_comb(self, t, start, end, pre_sum):
        m = self.L_scan_sort(t, start, end, pre_sum)
        self.D.append(self.c[np.argmax(m)])
        self.index.append(self.c[self.d-1])
        # print(self.index)
        self.c.clear()
        return np.max(m)

    def initialize(self):
        for n in range(1, self.N - self.M + 2):
            self.B[1, n] = 0
    
    def maximize(self):
        for m in range(2, self.M+1):
            for n in range(m, m + self.N - self.M + 1):
                L = np.empty(n-m+1)
                for h in range(m-1, n):
                    start = self.B[m-1, h]+1
                    pre_sum = self.statistic[m-1, h]
                    L[h-m+1] = self.max_comb(h, start, n, pre_sum)


                self.B[m, n] = np.argmax(L) + (m-1)
                self.statistic[m, n] = np.max(L)
                self.dimension[m][n] = self.D[np.argmax(L)]
                self.SortIndex[m][n] = copy.deepcopy(self.index)
                self.index.clear()
                self.D.clear()

    def detect_change_points(self):
        self.initialize()
        self.maximize()
    
    def backtrack(self):
        t_hat = []
        dim = []
        t_hat.append(self.N)
        for l in range(2, self.M+1)[::-1]:
            t_hat.append(self.B[l, t_hat[len(t_hat)-1]])
            dim.append(self.dimension[l][t_hat[len(t_hat)-2]])
        t_hat.append(self.B[1, t_hat[len(t_hat)-1]])

        t_hat.reverse()
        dim.reverse()

        return t_hat, dim


    def create_graph(self):
        plt.figure(figsize=(10,5),dpi=100)
        t, dimensions = self.backtrack()
        for i in range(len(t)-2):
            x = np.arange(t[i], t[i+2], 1)
            L = np.empty(len(x))
            for s in range(t[i], t[i+2]):
                L_s = np.empty(self.d+1)
                for p in range(1, self.d+1):
                    comb = common_function.Combination(self.d, p)
                    points = comb.combinationList()

                    l = np.empty(len(points))
                    onevec = common_function.OneVec(self.d)
                    for k in range(len(points)):
                        vec = onevec.get_2(points[k])
                        Z = self.Z_t(s, t[i], t[i+2])
                        Z = Z*vec
                        l[k] = (np.dot(Z,Z) - p)/(np.sqrt(2*p))
                    L_s[p-1] = np.max(l)
                L[s-t[i]] = np.max(L_s)
            plt.plot(x, L, label='segment'+str(i))
            plt.legend()

        plt.savefig('scan.pdf')
        plt.clf()

    def get_B(self):
        return self.B
    
    def get_dimension(self):
        return self.dimension
    
    def get_statistic(self):
        return self.statistic

    def get_sortIndex(self):
        return self.SortIndex
