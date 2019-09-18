# -*- coding: utf-8 -*-

# timewave
# --------
# timewave, a stochastic process evolution simulation engine in python.
# 
# Author:   sonntagsgesicht, based on a fork of Deutsche Postbank [pbrisk]
# Version:  0.6, copyright Wednesday, 18 September 2019
# Website:  https://github.com/sonntagsgesicht/timewave
# License:  Apache License 2.0 (see LICENSE file)


from pprint import pprint
from math import sqrt


class IndexMatrix(list):

    def __init__(self, *items):
        super(IndexMatrix, self).__init__()
        self._keys = items

    def get(self, k, d=None):
        i, j = k
        if isinstance(i, list):
            return [self.get((ii, j), d) for ii in i]
        elif isinstance(j, list):
            return [self.get((i, jj), d) for jj in j]
        else:
            return super(IndexMatrix, self).get(k, d)


def _fill_sparse_correlation(risk_factor_list, correlation):
    _correlation = dict() if correlation is None else dict(correlation)
    # fill sparse correlation matrix
    for rf_1 in risk_factor_list:
        for rf_2 in risk_factor_list:
            if (rf_1, rf_2) not in _correlation:
                if (rf_2, rf_1) in _correlation:
                    # stay symmetric
                    _correlation[rf_1, rf_2] = _correlation[rf_2, rf_1]
                else:
                    # populate with defaults
                    _correlation[rf_1, rf_2] = 1. if rf_1 == rf_2 else 0.
    return _correlation


def cholesky(A):
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = sqrt(A[i][i] - s) if (i == j) else (1.0 / L[j][j] * (A[i][j] - s))
    return L


def mmult(A, B):
    C = [[0.0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            C[i][j] = sum([A[i][k] * B[k][j] for k in range(len(B))])
    return C


def mtrans(A):
    return list(map(list, list(zip(*A))))


def munit(n):
    E = [[0.0] * n for _ in range(n)]
    for i in range(n):
        E[i][i] = 1.0
    return E


def mop(A, B, op):
    C = [[0.0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            if op == '+':
                C[i][j] = A[i][j] + B[i][j]
            elif op == '-':
                C[i][j] = A[i][j] - B[i][j]
    return C


def madd(A, B):
    return mop(A, B, '+')


def msub(A, B):
    return mop(A, B, '-')


def mdiag(v):
    C = [[0.0] * len(v) for _ in range(len(v))]
    for i in range(len(v)):
        C[i][i] = v[i]
    return C


def smult(s, A):
    return mmult(mdiag([s] * len(A)), A)


if __name__ == "__main__":

    m1 = [[25, 15, -5],
          [15, 18, 0],
          [-5, 0, 11]]
    c1 = cholesky(m1)
    pprint(c1, width=20)
    t1 = mtrans(c1)
    pprint(t1, width=25)
    s1 = mmult(c1, t1)
    pprint(s1, width=25)
    pprint(mdiag([1,2,3]), width=20)
    pprint((smult(10., munit(3))), width=20)
    print("")

    m2 = [[18, 22, 54, 42],
          [22, 70, 86, 62],
          [54, 86, 174, 134],
          [42, 62, 134, 106]]
    pprint(cholesky(m2), width=120)
