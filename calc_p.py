# coding: utf-8
"""
Main function
"""

import numpy as np
import sys
import os
import csv

from Core import scan_DP_sort
from Core import si_allvec_sort_sep


#----------------------------------------------------------------------
#変化点とp値をcsvで出力する関数
def out_put_result(t, dim, selective, M):
    """
    change_point, change_dim, selective_p
    """
    if not os.path.isdir("result"):
        os.mkdir("result")
    file_name = './result/result_set.csv'
    with open(file_name, 'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(t)
        writer.writerows(dim)
        writer.writerow(selective)
        writer.writerow([M])


#----------------------------------------------------------------------
#本体
if __name__ == '__main__':
    args = sys.argv
    if len(args) != 5:
        print('The number of arguments does not match')
        exit()
    datafile = args[1]
    sigmafile = args[2]
    xifile = args[3]
    M = int(args[4])

    X = np.loadtxt(datafile,delimiter=',')
    sigma = np.loadtxt(sigmafile,delimiter=',')
    xi = np.loadtxt(xifile,delimiter=',')
        
    scan = scan_DP_sort.Scan(X, M)
    t, dimensions = scan.backtrack()

    si = si_allvec_sort_sep.SI_all_vec(X, t, dimensions, scan.get_B(), scan.get_dimension(), scan.get_sortIndex(), sigma, xi)
    selective = si.inference()

    out_put_result(t, dimensions, selective, M)
            
