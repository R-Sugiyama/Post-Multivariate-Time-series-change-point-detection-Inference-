# coding: utf-8
"""
Commonly used functions and classes
"""

import numpy as np
import math
import copy
from mpmath import mp
mp.dps = 3000



#----------------------------------------------------------------------
#1ベクトルを作るクラス

class OneVec():
    def __init__(self, size):
        self.size = size

    def get(self, i, j=None):
        """
        Get the vector

        [Parameters]
        i <int> : start index of one (1≦i≦size)
        j <int> : end index of one (1≦j≦size)

        [Returns]
        numpy.ndarray
        """
        vec = np.zeros(self.size)

        if j is None:
            vec[i-1] = 1
        else:
            vec[i-1:j] = 1
        
        return vec

    def get_2(self, l):
        """
        Get the vector

        [Parameters]
        l [list] : index of projection

        [Returns]
        numpy.ndarray
        """
        vec = np.array([1 if i in(l) else 0 for i in range(self.size)])
        return vec


#----------------------------------------------------------------------
#1行列を作るクラス
class Diag():
    def __init__(self, n, d):
        self.n = n
        self.d = d #nは系列の長さ，dは系列数

    def get_diag(self, s, t):

        """
        Get the Matrix(nd×d)

        [Parameters]
        s <int> : start number of one (1≦s≦n)
        t <int> : end number of one (1≦t≦n)

        [Returns]
        numpy.ndarray
        """

        diag = np.zeros((self.n*self.d,self.d))

        for i in range(self.d):
            for j in range(s, t+1):
                diag[i*self.n + j - 1][i] = 1

        return diag

    def get_Pi(self, m):

        """
        Get the Diagonal matrix(nd x nd)

        [Parameters]
        m [list] : the dimension of Identity

        [Returns]
        numpy.ndarray
        """

        diag = np.zeros((self.n*self.d, self.n*self.d))
        for d in m:
            identity = np.identity(self.n)
            diag[d*self.n:d*self.n+self.n,d*self.n:d*self.n+self.n] = identity
        return diag

#----------------------------------------------------------------------
#切断正規分布の累積分布を計算する関数
def tn_cdf(x, intervals, mean=0, var=1):
  """
  CDF of a truncated normal distribution

  [Parameters]
    x <float> : Return the value at x
    intervals <list np.ndarray> : [L, U] or [[L1, U1], [L2, U2],...]
    mean <float>
    var <float>

  [Returns]
    float : CDF of TN
  """
  if isinstance(intervals, (list, tuple)):
    intervals = np.array(intervals)

  if len(intervals.shape) == 1:
    intervals = np.array([intervals,])

  n_intervals = len(intervals)  # number of intervals
  sd = math.sqrt(var)  # standard deviation

  # locate the interval that contains x
  for i in range(n_intervals):
    if intervals[i][0] <= x <= intervals[i][1]:
      x_index = i
      break
  else:
    raise ValueError(f'tn_cdf at x={x} is undefined.\nintervals={intervals}')

  norm_intervals = (intervals - mean) / sd  # normalized intervals

  # calculate the sum(delta) in order
  delta = [0,]
  for i in range(n_intervals):
    diff = mp.ncdf(norm_intervals[i][1]) - mp.ncdf(norm_intervals[i][0])
    delta.append(delta[-1] + diff)

  numerator = mp.ncdf((x-mean)/sd) - mp.ncdf(norm_intervals[x_index][0]) + delta[x_index]
  denominator = delta[-1]

  return float(numerator / denominator)

#----------------------------------------------------------------------

class QuadraticInterval():
  """
  Calculate the truncation intervals E(z)=∩{ατ^2+βτ+γ≦0} for quadratic
  selection events. τ is test statistic, and β,γ are function of c,z.

  [Constructor]
    tau <float> : Set tau=0 unless we consider E(z)=∩{ατ^2+κτ+λ≦0}.      κ,λ are function of c,x.
  """
  def __init__(self, tau=0, init_lower=0, init_upper=float('inf')):
    self.tau = tau
    self.lower = init_lower
    self.upper = init_upper
    self.concave_intervals = []

  def cut(self, a, b, c):
    """
    Truncate the interval with a quadratic inequality ατ^2+βτ+γ≦0

    [Parameters]
      a <float> : α
      b <float> : β or κ
      c <float> : γ or λ
    """
    if -1e-10 < a < 1e-10:
      a = 0
    if -1e-10 < b < 1e-10:
      b = 0
    if -1e-10 < c < 1e-10:
      c = 0

    if a == 0:
      if b == 0:
        return
      elif b < 0:
        self.lower = max(self.lower, -c/b + self.tau)
      elif b > 0:
        self.upper = min(self.upper, -c/b + self.tau)
    elif a > 0:
      disc = b**2 - 4*a*c  # discriminant
      self.lower = max(self.lower, (-b-math.sqrt(disc)) / (2*a) + self.tau)
      self.upper = min(self.upper, (-b+math.sqrt(disc)) / (2*a) + self.tau)
    else:
      disc = b**2 - 4*a*c  # discriminant
      if disc <= 0:  # no solution
        return

      lower = (-b+math.sqrt(disc)) / (2*a) + self.tau
      upper = (-b-math.sqrt(disc)) / (2*a) + self.tau

      # To reduce the calculation cost in _concave_cut,
      # truncate the interval if it's possible at this moment
      if lower < self.lower < upper < self.upper:
        self.lower = upper
      elif self.lower < lower < self.upper < upper:
        self.upper = lower
      elif self.lower < lower < upper < self.upper:
        # The interval is splitted into 2 parts at this moment
        self.concave_intervals.append((lower, upper))

  def _concave_cut(self):
    intervals = [[self.lower, self.upper],]

    for lower, upper in self.concave_intervals:
      if lower < intervals[0][0] < upper < intervals[-1][1]:
        # truncate the left side of the intervals
        for i in range(len(intervals)):
          if upper < intervals[i][0]:
            intervals = intervals[i:]
            break
          elif intervals[i][0] < upper < intervals[i][1]:
            intervals = intervals[i:]
            intervals[0][0] = upper
            break
      elif intervals[0][0] < lower < intervals[-1][1] < upper:
        # truncate the right side of the intervals
        for i in range(len(intervals)-1, -1, -1):
          if intervals[i][1] < lower:
            intervals = intervals[:i+1]
            break
          elif intervals[i][0] < lower < intervals[i][1]:
            intervals = intervals[:i+1]
            intervals[-1][1] = lower
            break                
      elif intervals[0][0] < lower < upper < intervals[-1][1]:
        # truncate the middle part of the intervals
        for i in range(len(intervals)):
          if upper < intervals[i][0]:
            right_intervals = intervals[i:]
            break
          elif intervals[i][0] < upper < intervals[i][1]:
            right_intervals = copy.deepcopy(intervals[i:])
            right_intervals[0][0] = upper
            break

        for i in range(len(intervals)-1, -1, -1):
          if intervals[i][1] < lower:
            left_intervals = intervals[:i+1]
            break
          elif intervals[i][0] < lower < intervals[i][1]:
            left_intervals = copy.deepcopy(intervals[:i+1])
            left_intervals[-1][1] = lower
            break

        intervals = left_intervals + right_intervals

    return np.array(intervals)

  def get(self):
    """
    Get the intervals

    [Returns]
      numpy.ndarray
    """
    return self._concave_cut()

#----------------------------------------------------------------------
#組み合わせ計算
class Combination():
    def __init__(self, n, p):
        self.data = [i for i in range(n)]
        self.p = p

    def combination(self, n, r):
        if(n < r):
            return 0
        return int(math.factorial(n) / (math.factorial(r) * math.factorial(n - r)))

    def getComginationIndex(self, length, r, lastIndex):
        result = []
        for i in range(r):
            if(len(lastIndex) == 0):
                result.append(i)
            elif(lastIndex[i] >= length - r + i):
                result = self.updateCombinationIndex(lastIndex, i - 1, False)
                break
            elif(i == r - 1):
                result = self.updateCombinationIndex(lastIndex, i, False)
        return result

    def combinationList(self):
        """
        Get the list


        [Returns]
        list = {A \subset {1,2,...,n} | #A = p}
        """
        length = len(self.data)
        total = self.combination(length, self.p)
        result = []
        lastIndex = []
        for i in range(total):
            lastIndex = self.getComginationIndex(length, self.p, lastIndex)
            result.append(self.getListElements(self.data, lastIndex))
        return result

    def combinationWithRepetition(self, n, r):
        return int(math.factorial(r + n - 1) / (math.factorial(r) * math.factorial(n - 1)))

    def updateCombinationIndex(self, lastIndex, start, repetition=False):
        result = lastIndex[:start]
        x = lastIndex[start] + 1
        for i in range(start, len(lastIndex)):
            result.append(x)
            if(repetition == False):
                x += 1
        return result

    def getListElements(self, lis, indices):
        """listからindicesで示される要素の組を返す"""
        return [lis[i] for i in indices]