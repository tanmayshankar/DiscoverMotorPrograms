from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import time
import pdb
import torch

epsilon = 1e-6

## 
#                          | C[m-1,n']
# C[m,n,n'] = d_mn + min   | C[m, n']
#                          | C[m-1, n, n']

# where, C[m, n] = \sum_n' p(n' | n) (C[m, n, n'])
def nested_tensor_list(sizes, device):
    if len(sizes) == 1:
        return [torch.as_tensor(0, device=device).float() for ix in range(sizes[0])]
    else:
        return [nested_tensor_list(sizes[1:], device) for ix in range(sizes[0])]


def nested_sparse_tensor_list(sizes, num_neighbors, device):
    if len(sizes) == 1:
        return [nested_tensor_list([num_neighbors[ix]], device) for ix in range(sizes[0])]
    else:
        return [nested_sparse_tensor_list(sizes[1:], num_neighbors, device) for ix in range(sizes[0])]
    


def stack_nested_tensor_list(tl):
    if torch.is_tensor(tl[0]):
        return torch.stack(tl)
    else:
        return torch.stack([stack_nested_tensor_list(t) for t in tl])


def compute_var_requirement(match_argmin, is_edge):
    M = match_argmin.shape[0]
    N = match_argmin.shape[1]
    is_req_C = torch.zeros(M, N)
    is_req_C_bar = torch.zeros(M, N, N)
    is_req_C[-1, :] = 1

    for m in range(M-1,-1,-1):
        for n in range(N-1,-1,-1):
            if is_req_C[m,n] > 0:
                is_req_C_bar[m,n,:] += is_edge[n, :].float()

            for n_bar in range(N-1,-1,-1):
                if is_req_C_bar[m, n, n_bar]:
                    if match_argmin[m,n,n_bar] == 0:
                        is_req_C[m-1, n_bar] += 1
                    elif match_argmin[m,n,n_bar] == 1:
                        is_req_C[m, n_bar] += 1
                    elif match_argmin[m,n,n_bar] == 2:
                        is_req_C_bar[m-1,n,n_bar] += 1

    return is_req_C_bar


def construct_transition_probs(next_s_probs, next_p_probs, seg_lengths):
    '''
    Args:
        next_s_probs: prob of continuing to next state in current primitive. should be 0 for last state in each prim
        next_p_probs: K probs for using next primitive. should be 0 for last one
        seg_lengths: max length of each primitive
    '''
    N = next_s_probs.shape[0]
    q = torch.zeros(N+1, N).to(next_s_probs.device)
    _end = N
    s = 0
    for p in range(len(seg_lengths)):
        seg_len = seg_lengths[p]
        pc_prim = next_p_probs[p]
        # pc_prim = 1
        s_prim_end = s + seg_len # next state if primitive ends
        for ix in range(seg_len):
            if ix == seg_len-1:
                p_ns = 0
            else:
                p_ns = next_s_probs[s]
                # p_ns = 1.0
            q[s+1, s] = q[s+1, s] + p_ns
            q[s_prim_end, s] = q[s_prim_end, s] + (1-p_ns)*pc_prim
            q[_end, s] = q[_end, s] + (1-p_ns)*(1-pc_prim)
            s += 1
    return q


def sample_ind(p):
    '''
    Args:
        p: N dim likelihood
    Returns:
        n ~ N, sampled from prob
    '''
    dist = torch.distributions.categorical.Categorical(probs=p)
    return dist.sample()


class ProbabilisticAlignmentLoss(torch.nn.Module):
    def __init__(self, p=2, dist_fn=None):
        '''
        Args:
            p: distance norm
            dist_fn: custom differentiable function to compute distance between states
        '''
        super(ProbabilisticAlignmentLoss, self).__init__()
        self.p = p
        self.dist_fn=dist_fn

    def compute_dists(self, x, y):
        '''
        Args:
            x: M X D sequence
            y: N X D sequence
        Returns:
            pwise_dist: M X N distance matrix
        '''
        M = x.shape[0]
        N = y.shape[0]

        if self.dist_fn is None:
            diff = x.view(x.shape[0], 1, -1) - y.view(1, y.shape[0], -1)
            pwise_dist = torch.norm(diff, p=self.p, dim=2)
        else:
            pwise_dist = []
            for m in range(M):
                for n in range(N):
                    pwise_dist.append(self.dist_fn(x[m], y[n]))
            pwise_dist = torch.stack(pwise_dist).view(M, N)
        return pwise_dist

    def compute_reachability(self, q):
        '''
        Args:
            q: N+1 X N matrix, with q(n, n') = prob of jumping to n from n'. N+1 == end.
        Returns:
            p: (N) likelihood of visiting n
            p_back: N+1 X N, p(n, n'): given currently at n, prob of having been at n' in previous step
        '''
        N = q.shape[1]

        p = torch.zeros(N+1, device=q.device)
        p[0] = 1

        for n_bar in range(0, N):
            p = p + p[n_bar]*q[:, n_bar]

        p_back = p[0:N].view(1, N)*q/(p.view(N+1, 1) + epsilon)
        return p, p_back

    def compute_expected_cost_matrix(self, dists, p_back):
        '''
        Args:
            dists: M X N dists between x[m] and y[n]
            p_back: N+1 X N, p(n, n'): given currently at n, prob of having been at n' in previous step
        Computes:
            C (M X N): C_bar[m,n], expected loss between (x_0, x_1 .. x_m), (y_0, .. until .. y_n)
            C_bar (M X N X N): C_bar[m,n,n'], expected loss between (x_0, x_1 .. x_m), (y_0, .. until .. y_n', y_n)
            match_argmin (M X N X N): matches[m,n,n'] \in {0,1,2}
        '''

        M, N = dists.shape

        match_argmin = np.zeros((M, N, N))

        is_edge = p_back.detach() > 0
        ancestors = []
        num_ancestors = []

        for n in range(N):
            a_n = []
            for n_bar in range(n):
                if is_edge[n, n_bar]:
                    a_n.append(n_bar)
            ancestors.append(a_n)
            num_ancestors.append(len(a_n))

        C = nested_tensor_list([M, N], device=p_back.device)
        C_bar = nested_sparse_tensor_list([M, N], num_ancestors, device=p_back.device)

        for m in range(M):
            if m == 0:
                C[m][0] = dists[m, 0]
            else:
                C[m][0] = dists[m, 0] + C[m-1][0]

        # C = torch.zeros(M, N, device=p_back.device)
        # C_bar = C.unsqueeze(2).repeat(1, 1, N)
        # C[:, 0] = torch.cumsum(dists[:, 0], dim=0)

        for m in range(M):
            for n in range(1, N):
                d_mn = dists[m, n]
                for n_bar_ind, n_bar in enumerate(ancestors[n]):
                    if is_edge[n, n_bar]:
                        c_min = C[m][n_bar]
                        match = 1

                        if m > 0 and C[m-1][n_bar] < c_min:
                            c_min = C[m-1][n_bar]
                            match = 0

                        if m > 0 and C_bar[m-1][n][n_bar_ind] < c_min:
                            c_min = C_bar[m-1][n][n_bar_ind]
                            match = 2

                        # if match == 0:
                        #     C_bar[m][n][n_bar] = d_mn + C[m-1][n_bar]
                        # elif match == 1:
                        #     C_bar[m][n][n_bar] = d_mn + C[m][n_bar]
                        # elif match == 2:
                        #     C_bar[m][n][n_bar] = d_mn + C_bar[m-1][n][n_bar]
  
                        C_bar[m][n][n_bar_ind] = d_mn + c_min
                        match_argmin[m,n,n_bar] = match

                        C[m][n] = C[m][n] + p_back[n,n_bar]*C_bar[m][n][n_bar_ind]

                #print(p_back[n,:].shape)
                #print(C_bar[m,n,:].shape)
                #print(C.shape)

                # C[m][n] = torch.sum(p_back[n,:]*C_bar[m,n,:])

        # self.C_bar = stack_nested_tensor_list(C_bar)
        self.C_bar = C_bar
        self.C = stack_nested_tensor_list(C)

        # self.C_bar = C_bar
        # self.C = C

        self.match_argmin = match_argmin

    def forward(self, x, y, q):
        '''
        Args:
            x: M X D sequence
            y: N X D sequence
            q: N+1 X N matrix, with q(n, n') = prob of jumping to n from n'. N+1 == end
        Returns:
            cost: expected matching cost
        '''
        t_init = time.time()

        dists = self.compute_dists(x, y)
        _, p_back = self.compute_reachability(q)
        self.compute_expected_cost_matrix(dists, p_back)
        cost = torch.sum(p_back[-1, :]*self.C[-1, :])
        return cost

    def sample(self, y, q):
        '''
        Args:
            y: N X D sequence
            q: N+1 X N matrix, with q(n, n') = prob of jumping to n from n'. N+1 == end
        Returns:
            y_s: L X D sampled sequence
            inds: L indices of y that were sampled
        '''
        y_s = []
        inds = []
        probs = []
        n = 0
        N = y.shape[0]
        while n < N:
            y_s.append(y[n])
            inds.append(n)
            n_bar = n
            n = sample_ind(q[:, n_bar])
            probs.append(q[n, n_bar])

        return torch.stack(y_s), torch.Tensor(inds), torch.stack(probs)

    def recover_alignment(self, inds):
        '''
        Args:
            inds: L indices of y that were sampled
        Return:
            match_x, match_y: M and L indices indicating x,y matches
        '''
        M = self.match_argmin.shape[0]
        match_x = []
        match_y = []
        m = M-1
        ind = len(inds)-1
        while (m > 0) or (ind > 0):
            n = inds[ind]
            if n > 0:
                n_bar = inds[ind-1]
                match = self.match_argmin[m,n,n_bar]
            else:
                match = 2 # i.e. reduce m

            if match == 0:
                match_x = [ind] + match_x
                match_y = [m] + match_y
                m -= 1
                ind -= 1

            elif match == 1:                
                match_y = [m] + match_y
                ind -= 1

            elif match == 2:
                match_x = [ind] + match_x
                m -= 1
