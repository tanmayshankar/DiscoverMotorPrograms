from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pdb
import numpy as np
import torch

from ..utils import alignment


class StepLoss(torch.nn.Module):
    def __init__(self, dist_fn=None, hinge=0.1):
        '''
        Args:
            p: distance norm
            dist_fn: custom differentiable function to compute distance between states
        '''
        super(StepLoss, self).__init__()
        self.hinge = hinge
        self.dist_fn=dist_fn

    def forward(self, x):
        x_seq = torch.cat((x[[0]], x), dim=0)
        dx = torch.norm(x - x_seq[0:x.shape[0]], dim=-1)
        # return torch.nn.functional.relu(dx-self.hinge).mean()
        return torch.nn.functional.relu(dx-self.hinge).sum()


class ConstantVelocityPrior(torch.nn.Module):
    def __init__(self):
        '''
        '''
        super(ConstantVelocityPrior, self).__init__()

    def forward(self, x):
        '''
        Args:
            x: T X F trajectory.
        Returns:
            loss: Loss evaluating how well the subtrajectory fits a constant velocity prior
                loss = min _{x_bar, delta} \sum{t=0 to T-1} (x_t - x_bar - delta*(t - t_bar))^2, where t_bar = (T-1)/2
                here, the optimal x_bar = mean(xs) and delta = \sum_t (t - t_bar)*(x_t - x_bar) / \sum_t (t - t_bar)^2
        '''
        x_bar = torch.mean(x, dim=0, keepdim=True)
        T = x.shape[0]

        if T < 3:
            return torch.mean(x*0)

        t_bar = T/2 - 0.5
        ts  = torch.linspace(0, T-1, T, device=x.device, dtype=x.dtype).view(T, 1)
        t_diffs = ts - t_bar
        delta_xs = x - x_bar
        ## delta: 1 X F
        delta = torch.sum(delta_xs*t_diffs, dim=0, keepdim=True)/torch.sum(torch.pow(t_diffs, 2), dim=0, keepdim=True)
        loss = torch.mean(torch.pow(x - x_bar - t_diffs*delta, 2))
        return loss


class TransitionAlignmentLoss(torch.nn.Module):
    def __init__(self, p=2, trans_wt=1, dist_fn=None, diff_dist_fn=None):
        '''
        Args:
            p: distance norm
            trans_wt: weight of transitions
            dist_fn: custom differentiable function to compute distance between states
            diff_dist_fn: custom differentiable function to compute distance between transitions
        '''
        super(TransitionAlignmentLoss, self).__init__()
        self.p = p
        self.trans_wt = trans_wt
        self.dist_fn = dist_fn
        self.diff_dist_fn = diff_dist_fn

    def forward(self, x, y):
        '''
        Args:
            x, y: M X F and N X F trajectories
        '''
        x_seq = torch.cat((x[[0]], x), dim=0)
        y_seq = torch.cat((y[[0]], y), dim=0)

        dx = x - x_seq[0:x.shape[0]]
        dy = y - y_seq[0:y.shape[0]]
        
        M = x.shape[0]
        N = y.shape[0]

        ## reshape
        with torch.no_grad():
            if self.dist_fn is None:
                diff = x.view(x.shape[0], 1, -1) - y.view(1, y.shape[0], -1)
                pwise_dist = torch.norm(diff, p=self.p, dim=2)
            else:
                pwise_dist = []
                for m in range(M):
                    for n in range(N):
                        pwise_dist.append(self.dist_fn(x[m], y[n]))
                pwise_dist = torch.stack(pwise_dist).view(M, N)

            if self.diff_dist_fn is None:
                pwise_t_dist = torch.norm(
                    dx.view(dx.shape[0], 1, -1) - dy.view(1, dy.shape[0], -1),
                    p=self.p, dim=2
                )*self.trans_wt
            else:
                pwise_t_dist = []
                for m in range(M):
                    for n in range(N):
                        pwise_t_dist.append(self.diff_dist_fn(dx[m], dy[n]))
                pwise_t_dist = torch.stack(pwise_t_dist).view(M, N)

            ## Compute best alignment
            pcm = pwise_dist.cpu().numpy()
            tcm = pwise_t_dist.cpu().numpy()

            acm = alignment.compute_alignment_cost_matrix(pcm, tcm=tcm)
            inds_x, inds_y, (unmatched_x, unmatched_y) = alignment.recover_matching(pcm, acm, tcm=tcm)
            self._soln_inds = (inds_x, inds_y)

        ## Compute matching cost
        cost = 0
        if self.dist_fn == None:
            df = lambda a, b : torch.norm(a-b, p=self.p)
        else:
            df = self.dist_fn

        if self.diff_dist_fn == None:
            df_diff = lambda a, b : torch.norm(a-b, p=self.p)
        else:
            df_diff = self.diff_dist_fn

        for m in range(inds_x.shape[0]):
            #cost += pwise_dist[m, inds_x[m]]
            if unmatched_x[m]:
                cost += df(x[m], y[inds_x[m]]) + df_diff(dx[m], dy[0])
            else:
                #print(dx[m])
                #print(inds_x[m])
                #print(dy[inds_x[m]])
                cost += df(x[m], y[inds_x[m]]) + df_diff(dx[m], dy[inds_x[m]])

        for n in range(inds_y.shape[0]):
            if unmatched_y[n]:
                # the matched ys will have already been used in loop above
                cost += df(x[inds_y[n]], y[n]) + df_diff(dx[0], dy[n])

        return cost


class AlignmentLoss(torch.nn.Module):
    def __init__(self, p=2, dist_fn=None):
        '''
        Args:
            p: distance norm
            dist_fn: custom differentiable function to compute distance between states
        '''
        super(AlignmentLoss, self).__init__()
        self.p = p
        self.dist_fn=dist_fn

    def forward(self, x, y):
        '''
        Args:
            x, y: M X F and N X F trajectories
        '''
        # print(x.shape, y.shape)
        M = x.shape[0]
        N = y.shape[0]

        with torch.no_grad():
            if self.dist_fn is None:
                diff = x.view(x.shape[0], 1, -1) - y.view(1, y.shape[0], -1)
                pwise_dist = torch.norm(diff, p=self.p, dim=2)
            else:
                pwise_dist = []
                for m in range(M):
                    for n in range(N):
                        pwise_dist.append(self.dist_fn(x[m], y[n]))
                pwise_dist = torch.stack(pwise_dist).view(M, N)
        
            ## Compute best alignment
            pcm = pwise_dist.detach().cpu().numpy()
            
            acm = alignment.compute_alignment_cost_matrix(pcm)
            inds_x, inds_y, (unmatched_x, unmatched_y) = alignment.recover_matching(pcm, acm)

            # save solution as it's possibly useful for debugging
            self._soln_inds = (inds_x, inds_y)

        ## Compute matching cost
        cost = 0
        if self.dist_fn == None:
            df = lambda a, b : torch.norm(a-b, p=self.p)
        else:
            df = self.dist_fn

        for m in range(inds_x.shape[0]):
            #cost += pwise_dist[m, inds_x[m]]
            cost += df(x[m], y[inds_x[m]])

        for n in range(inds_y.shape[0]):
            if unmatched_y[n]:
                # the matched ys will have already been used in loop above
                # cost += pwise_dist[inds_y[n], n]
                cost += df(x[inds_y[n]], y[n])

        return cost


if __name__ == '__main__':
    x = np.array([[1],[3],[5]])
    y = np.array([[1],[3]])

    #x = np.random.randint(0,10,(100, 10))
    #y = np.random.randint(0,10,(50, 10))

    x = torch.from_numpy(x).float().cuda()
    y = torch.from_numpy(y).float().cuda()

    loss_func = AlignmentLoss()

    t_init = time.time()
    cost = loss_func(x, y)
    print(time.time() - t_init)

    print(cost)
