# coding: utf-8
"""
Main function
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os

from Core import multidimensional_data
from Core import scan_DP_sort
from Core import si_allvec_sort_sep
from Core import common_function
# import multidimensional_data
# import scan_DP_sort
# import si_allvec_sort_sep

#----------------------------------------------------------------------
#初期化
n = 30 #データ数
M = 3 #分割セグメント数
sigma = 1
test = 10000

#----------------------------------------------------------------------
#FPRの評価の図
def create_figure(selective, naive, identity):
    x = identity
    plt.ylim([-0.05, 1.05])
    plt.yticks([(i+1)*0.2 for i in range(5)])
    plt.xticks([(i+2) for i in range(5)])
    plt.ylabel('FPR')
    plt.xlabel('Number of Series')
    plt.plot(x, naive, label='naive', marker='o')
    plt.plot(x, selective, label='selective', marker='o')
    plt.hlines([0.05], 1.75, 6.25, color='g', linestyles='dashed')

    plt.legend()
    plt.savefig('./result/result_FPR_dim.pdf')
    plt.clf()

#----------------------------------------------------------------------
#本体
if __name__ == '__main__':
    selective_FPR = np.array([])
    naive_FPR = np.array([])
    dim = [2,3,4,5,6]
    for d in dim:
        selective_p = np.array([])
        naive_p = np.array([])
        change_point = [[0, n]for j in range(d)]
        change_mean = [[i] for i in range(d)]
        for i in range(test):
            print('count'+str(i))
            data = multidimensional_data.MultiData(n, d)
            X = data.create_data(sigma, change_point, change_mean)

            scan = scan_DP_sort.Scan(X, M)
            t, dimensions = scan.backtrack()
            print(t)
            print(dimensions)

            si = si_allvec_sort_sep.SI_all_vec(X, t, dimensions, scan.get_B(), scan.get_dimension(), scan.get_sortIndex(), sigma)
            selective, naive = si.inference()

            print(selective)
            print(naive)

            selective_p = np.append(selective_p, np.array([random.choice(selective)]))
            naive_p = np.append(naive_p, np.array([random.choice(naive)]))
        selective_FPR = np.append(selective_FPR, len(selective_p[selective_p < 0.05])/len(selective_p))
        naive_FPR = np.append(naive_FPR, len(naive_p[naive_p < 0.05])/len(naive_p))
    if not os.path.isdir("result"):
        os.mkdir("result")
    np.savetxt('./result/FPR_dim.csv', np.array([selective_FPR, naive_FPR]), delimiter=',')
    create_figure(selective_FPR, naive_FPR, dim)
