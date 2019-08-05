"""
Utils related to geometry like projection
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def hamilton_product(qa, qb):
    """Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]
    
    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]
    
    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0
    
    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def quat_rotate(X, q):
    """Rotate points by quaternions.
    Args:
        X: B X N X 3 points
        q: B X 4 quaternions
    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    q = torch.unsqueeze(q, 1)*ones_x

    q_conj = torch.cat([ q[:, :, [0]] , -1*q[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, [0]]*0, X ], dim=-1)
    
    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]


def quat_conjugate(q):
    q_real, q_img = torch.split(q, [1,3], dim=-1)
    return torch.cat([-1*q_real, q_img], dim=-1)


def quat_dist(qa, qb):
    """Rotate points by quaternions.
    Args:
        q1: ... X 4 quaternions
        q2: ... X 4 quaternions
    Returns:
        dist: B X N (distance between quaternions)
    q_diff = multiply(qa, conjugate(qb))
    dist = 2*cos_inv(abs(q_diff[0]))
    In this case, q_diff[0] can be shown to be 
        qa_0*qb_0 + qa_1*qb_1 + qa_2*qb_2 + qa_3*qb_3
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/
    """
    q_cos_diff = torch.abs(torch.sum(qa * qb, dim=-1))
    # dist = 2*torch.acos(q_cos_diff)
    dist = 1 - (q_cos_diff)
    return dist
