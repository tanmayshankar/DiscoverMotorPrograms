from __future__ import division
from __future__ import print_function
import math
import numpy as np
import time


def compute_alignment_cost_matrix(pcm, tcm=None):
    '''
    Args:
        pairwise_cost_matrix (pcm): (M X N) cost matrix for matching cost of x[m] with y[n]
        [optional] transition_cost_matrix (tcm): (M X N) transition matching cost of x[m-1]->x[m] with
            y[n-1] -> y[n]. tcm[0,:] and tcm[:,0] use initial transition as 0.
    Returns:
        alignment_cost_matrix (acm): (M+1 X N+1) cost matrix for matching cost of x[m:] with y[n:]
            cost of matching x[M:] with y[n:] = \sum_j delta(x_{M-1}, y_j)
    '''
    M, N = pcm.shape
    acm = np.zeros((M+1, N+1))
    use_tcm = (tcm is not None)
    
    # fill base case: matching x[M:] (empty) with y[n:] i.e. all elems in y match with x[-1]
    for n in range(N-1,-1,-1):
        acm[M, n] = pcm[M-1, n] + acm[M, n+1] + (tcm[0, n] if use_tcm else 0)

    # fill base case: matching x[m:] (empty) with Y[N:](empty) i.e. all elems in x match with y[-1]
    for m in range(M-1,-1,-1):
        acm[m, N] = pcm[m, N-1] + acm[m+1, N] + (tcm[m, 0] if use_tcm else 0)
    
    # traverse matrix diagonally
    for mn_sum in range(M+N-2, -1, -1):
        m_min = max(0, mn_sum - (N-1))
        m_max = min(M-1, mn_sum)

        for m in range(m_min, m_max+1):
            n = mn_sum - m

            # cost = math.inf
            cost = float('inf')

            c_1 = acm[m+1, n+1] + pcm[m, n] + (tcm[m, n] if use_tcm else 0)
            cost = min(cost, c_1)

            if m > 0:
                c_2 = acm[m, n+1] + pcm[m-1, n] + (tcm[0, n] if use_tcm else 0)
                cost = min(cost, c_2)

            if n > 0:
                c_3 = acm[m+1, n] + pcm[m, n-1] + (tcm[m, 0] if use_tcm else 0)
                cost = min(cost, c_3)

            acm[m,n] = cost

    return acm


def recover_matching(pcm, acm, tcm=None):
    '''
    Args:
        pairwise_cost_matrix (pcm): (M X N) cost matrix for matching cost of x[m] with y[n]
        alignment_cost_matrix (acm): (M+1 X N+1) cumulative cost matrix
        [optional] transition_cost_matrix (tcm): (M X N) transition matching cost of x[m-1]->x[m] with
            y[n-1] -> y[n]. tcm[0,:] and tcm[:,0] use initial transition as 0.
    Returns:
        match_m: [M] index in [0, N-1] indicating match in other string
        match_n: [N] index in [0, M-1] indicating match in other string
        (skipped_m, skipped_n): binary variables indicating whether a variable was skipped
    '''
    M, N = pcm.shape
    use_tcm = (tcm is not None)

    match_m = np.zeros(M).astype(np.int32)
    match_n = np.zeros(N).astype(np.int32)

    skipped_m = np.zeros(M)
    skipped_n = np.zeros(N)

    m, n = 0, 0

    while not (m == M and n == N):
        if (m < M) and (n < N) and (acm[m,n] == acm[m+1, n+1] + pcm[m, n] + (tcm[m, n] if use_tcm else 0)):
            match_m[m] = n
            match_n[n] = m
            m += 1
            n += 1

        elif (n < N) and (m > 0) and (acm[m,n] == acm[m, n+1] + pcm[m-1, n] + (tcm[0, n] if use_tcm else 0)):
            match_n[n] = m-1
            skipped_n[n] = 1
            n += 1

        elif (m < M) and (n > 0) and (acm[m,n] == acm[m+1, n] + pcm[m, n-1] + (tcm[m, 0] if use_tcm else 0)):
            match_m[m] = n-1
            skipped_m[m] = 1
            m += 1

        else:
            raise Exception
    
    return match_m, match_n, (skipped_m, skipped_n)


def aligned_seq(x, y, match_m, match_n):
    a_x = []
    a_y = []
    m, n = 0, 0
    M, N = match_m.shape[0], match_n.shape[0]

    while not (m == M and n == N):
        if (m < M) and (n < N) and match_m[m] == n and match_n[n] == m:
            a_x.append(x[m])
            a_y.append(y[n])
            m += 1
            n += 1
        elif (n < N) and (m > 0) and match_n[n] == m-1:
            a_x.append(x[m-1])
            a_y.append(y[n])
            n += 1
        elif (m < M) and (n > 0) and match_m[m] == n-1:
            a_x.append(x[m])
            a_y.append(y[n-1])
            m += 1
        else:
            raise Exception

    return a_x, a_y



if __name__ == '__main__':
    # x = np.array([1,3,5])
    #y = np.array([1,3])
    x = np.random.randint(0,10,100)
    y = np.random.randint(0,10,50)

    pcm = np.zeros((x.shape[0], y.shape[0]))
    tcm = np.zeros((x.shape[0], y.shape[0]))

    x_seq = np.concatenate((x[[0]], x))
    y_seq = np.concatenate((y[[0]], y))

    dx = x_seq[1:] - x_seq[0:-1]
    dy = y_seq[1:] - y_seq[0:-1]

    #x = 'ACGCTGAT'
    #y = 'CAGCTAT'
    #pcm = np.zeros((len(x), len(y)))

    t1 = time.time()
    for m in range(pcm.shape[0]):
        for n in range(pcm.shape[1]):
            pcm[m,n] = abs(x[m] - y[n])
            tcm[m,n] = abs(dx[m] - dy[n])
            # pcm[m,n] = int(x[m] != y[n])

    acm = compute_alignment_cost_matrix(pcm, tcm=tcm)
    # print(acm)

    match_m, match_n, _ = recover_matching(pcm, acm, tcm=tcm)
    # print(match_m, match_n)

    a_x, a_y = aligned_seq(x, y, match_m, match_n)
    #print(x)
    #print(y)
    print(time.time()-t1)

    print(a_x)
    print(a_y)