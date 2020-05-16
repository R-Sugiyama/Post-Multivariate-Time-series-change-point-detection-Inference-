# coding: utf-8
"""
plot result
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

#----------------------------------------------------------------------
#選択された系列と変化点のグラフ
def create_graph(X, t, dimensions, selective, split):
    #グラフを見やすくするハイパーパラメータ
    shift = 4
    #データを見やすくするために標準化を行う
    X = np.array([(X[l,:] - np.mean(X[l,:]))/(np.std(X[l,:])) for l in range(split)])
    X = np.array([X[l,:] + l*shift for l in range(np.shape(X)[0])])

    plt.figure(figsize=(10,5),dpi=100)
    plt.xlabel('length',fontsize=20)
    plt.yticks([])
    for i in range(split):
        plt.plot(X[i], label='%dth_data'%(i+1), linewidth=3)

    alpha = 0.05 / len(selective)
    count = 0

    for i in range(1, len(t)-1):
        plt.vlines(t[i]-1, np.max(X)+(np.max(X)-np.min(X))*0.05, np.min(X)-(np.max(X)-np.min(X))*0.05, linestyle='dashed',linewidth=2.5)
        for j in dimensions[i-1]:
            if selective[count] < alpha:
                c = 'r'
            else:
                c = 'b'
            plt.plot([t[i]-1], [X[j][t[i]-1]], marker='o', markersize=10, color=c)
            plt.text(t[i]-1+0.5, X[j][t[i]-1]-0.2, str(round(selective[count],3)),fontsize=20)
            count += 1
    plt.tight_layout()
    plt.legend()
    plt.tick_params(labelsize=20)
    plt.savefig('./result/display_result.pdf')
    plt.close()

#----------------------------------------------------------------------
#実行
if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print('The number of arguments does not match')
        exit()
    datafile = args[1]

    file_name = './result/result_set.csv'
    with open(file_name) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        M = int(l[-1][0])
        l = [[float(r) if (i == M) else int(r) for r in l[i]] for i in range(len(l))]
    t = l[0]
    dim = [l[i] for i in range(1, M)]
    selective = l[M]

    X = np.loadtxt(datafile,delimiter=',')
    create_graph(X, t, dim, selective, M)
    