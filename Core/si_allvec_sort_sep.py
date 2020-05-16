import numpy as np
from scipy.stats import norm

from . import common_function

#----------------------------------------------------------------------
#SIを行うクラス
class SI_all_vec():
    def __init__(self, X, t, result_dim, B, dimension, sortIndex, sigma, Sigma):
        self.X = X
        self.vec_X = self.X.flatten()
        self.t = t
        self.result_dim = result_dim
        self.B = B
        self.dimension = dimension
        self.SortIndex = sortIndex
        self.N = np.shape(X)[1]
        self.d = np.shape(X)[0]
        self.M = np.shape(B)[0]-1
        self.cov = np.kron(sigma, Sigma)
        

        """
        input
            vec_X -> vectrized date
            t -> change_points
            dimension -> change_series sets
            n -> series length
            k -> a number of serieses
            statistic-> statistic matrix
            sigma -> date's variance
            Sigma -> covariance
        """

    def backtrack(self, n, m):
        """
        input
        n(int) end of the problem
        m(int) the number of split segment

        output
        t change points
        dim change series
        """
        t = []
        dim = []
        t.append(n)
        for l in range(2, m+1)[::-1]:
            t.append(self.B[l, t[len(t)-1]])
            dim.append(self.dimension[l][t[len(t)-2]])
        t.append(self.B[1, t[len(t)-1]])

        t.reverse()
        dim.reverse()

        return t, dim


    def inference(self):
        """
            output p_value
        """

        onevec = common_function.OneVec(self.N)

        quadra_cut = []
        eta_t = []
        c = []
        z = []

        for i in range(1, len(self.t)-1):
            for k in self.result_dim[i-1]:
                eta_k = np.zeros(self.N*self.d)
                eta = onevec.get(self.t[i-1]+1, self.t[i])/(self.t[i]-self.t[i-1]) - onevec.get(self.t[i]+1, self.t[i+1])/(self.t[i+1]-self.t[i])
                eta = np.sign(np.dot(eta, self.X[k,:]))*eta
                eta_k[k*self.N:(k+1)*self.N] = eta
                eta_t.append(eta_k)
                c.append(np.reciprocal(np.dot(np.dot(eta_k,self.cov),eta_k)) * np.dot(self.cov,eta_k))
                z.append(np.dot(np.identity(self.d*self.N)-np.outer(c[len(c)-1],eta_k),self.vec_X))
                quadra_cut.append(common_function.QuadraticInterval())


        for m in range(2, self.M+1):
            for n in range(m, m+self.N-self.M+1):
                t_hat, m_hat = self.backtrack(n,m)
                a_max, b_max = self.get_A_b_sum(t_hat, m_hat)
                for h in range(m-1,n):
                    #差の大きさによるソートのイベント Z(index) >= Z(index+1)
                    # for index in range(len(self.SortIndex[m][n][h-m+1])-1):
                    #     a_left = [self.get_Z(h, self.SortIndex[m][n][h-m+1][index], self.B[m-1,h], n)]
                    #     a_right = [self.get_Z(h, self.SortIndex[m][n][h-m+1][index+1], self.B[m-1,h], n)]

                    #     #区間の切断
                    #     for (c_, z_, quadra) in zip(c,z,quadra_cut):
                    #         alpha = self.calc_alpha(a_left, a_right, c_)
                    #         beta = self.calc_beta(a_left, a_right, z_, c_)
                    #         gamma = self.calc_gamma(a_left, a_right, z_, 0)
                    #         quadra.cut(alpha, beta, gamma)

                    t_hat, m_hat = self.backtrack(h, m-1)
                    a_sum, b_sum = self.get_A_b_sum(t_hat, m_hat)
                    # #各変化点で一番大きい統計量を選ぶイベント
                    for p in range(1, self.d+1):
                        m_hat = np.array(self.SortIndex[m][n][h-m+1])[:p]
                        a_h, b_h = self.get_A_b(h, m_hat, self.B[m-1,h], n)

                        a = a_sum + [a_h]

                        b = b_max - (b_sum+b_h)
                        
                        #区間の切断
                        for (c_, z_, quadra) in zip(c,z,quadra_cut):
                            alpha = self.calc_alpha(a_max, a, c_)
                            beta = self.calc_beta(a_max, a, z_, c_)
                            gamma = self.calc_gamma(a_max, a, z_, b)
                            quadra.cut(alpha,beta,gamma)

        quadra_interval = [item.get() for item in quadra_cut]
        selective_p_values = []
        for index, interval in enumerate(quadra_interval):
            selective_p_values.append(1 - common_function.tn_cdf(np.dot(eta_t[index],self.vec_X), interval, var=np.dot(np.dot(eta_t[index], self.cov),eta_t[index])))

        return selective_p_values
    
    #Zを表す行列の作成
    def get_Z(self, t, dim, start, end):

        onevec = common_function.OneVec(self.N*self.d)
        a = []
        vec_left = onevec.get(dim*self.N+start+1, dim*self.N+t)/(t-start)
        vec_right = onevec.get(dim*self.N+t+1, dim*self.N+end)/(end-t)
        vec = vec_left - vec_right
        vec = vec*np.sqrt(((t-start)*(end - t))/((end-start)))
        a.append(vec)

        return a

    #統計量の和を表す行列の作成
    def get_A_b_sum(self, t, m):
        b = 0
        a = []
        for i in range(1, len(t)-1):
            p = len(m[i-1])
            b += (np.sqrt(2*p))/2

            a_i = []
            onevec = common_function.OneVec(self.N*self.d)
            for dim in m[i-1]:
                vec_left = onevec.get(dim*self.N+t[i-1]+1, dim*self.N+t[i])/(t[i]-t[i-1])
                vec_right = onevec.get(dim*self.N+t[i]+1, dim*self.N+t[i+1])/(t[i+1]-t[i])
                vec = vec_left - vec_right
                vec = vec*np.sqrt(((t[i]-t[i-1])*(t[i+1] - t[i]))/((t[i+1]-t[i-1])))
                a_i.append(vec/np.sqrt(np.sqrt(2*p)))
            a.append(a_i)
        return a, b

    #統計量を表す行列の作成
    def get_A_b(self, t, m, start, end):
        p = len(m)
        b = (np.sqrt(2*p))/2

        onevec = common_function.OneVec(self.N*self.d)
        a = []
        for dim in m:
            vec_left = onevec.get(dim*self.N+start+1, dim*self.N+t)/(t-start)
            vec_right = onevec.get(dim*self.N+t+1, dim*self.N+end)/(end-t)
            vec = vec_left - vec_right
            vec = vec*np.sqrt(((t-start)*(end - t))/((end-start)*np.sqrt(2*p)))
            a.append(vec)

        return a, b

    def calc_alpha(self, a_max, a_sum, c):
        alpha = 0
        for dim in a_sum:
            for vec in dim:
                alpha += np.dot(c, vec)**2
        for dim in a_max:
            for vec in dim:
                alpha -= np.dot(c, vec)**2

        return alpha

    def calc_beta(self, a_max, a_sum, z, c):
        beta = 0
        for dim in a_sum:
            for vec in dim:
                beta += 2*np.dot(z, vec)*np.dot(c, vec)
        
        for dim in a_max:
            for vec in dim:
                beta -= 2*np.dot(z, vec)*np.dot(c, vec)

        return beta

    def calc_gamma(self, a_max, a_sum, z, b):
        gamma = 0
        for dim in a_sum:
            for vec in dim:
                gamma += np.dot(z, vec)**2
        for dim in a_max:
            for vec in dim:
                gamma -= np.dot(z, vec)**2

        return gamma + b