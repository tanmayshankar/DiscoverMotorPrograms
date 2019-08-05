"""
Script for training a simple inverse model using a regression loss.
Sample usage:
    python -m sfd.experiments.abstraction.random_walks --plot_scalars --display_visuals --display_freq=2000 --name=rw_known_zlen
    
    # With Transformer, with 3-6 segments in data, with prior loss.
    python -m sfd.experiments.abstraction.random_walks --plot_scalars --display_visuals --display_freq=2000 --name=T3_RW_varnseg36_pwt10 --n_segments_min=3 --n_segments_max=6 --prior_loss_wt=10 --transformer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from absl import app
from absl import flags

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torchvision

from ...DataLoaders import RandomWalks
from ...nnutils import train_utils
from ...nnutils import seq_alignment
from . import abstraction_utils as abs_util

opts = flags.FLAGS

flags.DEFINE_integer('n_state', 2, 'Dimension of state space')
flags.DEFINE_float('align_loss_wt', 1, 'Weight or alignment loss')
flags.DEFINE_float('sf_loss_wt', 0.1, 'Weight of pseudo loss for SF estimator')
flags.DEFINE_boolean('trans_align_loss', False, 'Use transition alignment loss instead of basic alignment loss')
flags.DEFINE_float('prior_loss_wt', 0, 'Weight for prior loss on each subsequence')
flags.DEFINE_float('parsimony_loss_wt', 0.01, 'Additional cost for using each primitive')
flags.DEFINE_boolean('transformer', False, 'Whether or not to use the transformer model (or default recurrent encoder decoder).')

class PrimitiveDiscoveryTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        # Now using transformer if we want.
        if self.opts.transformer:
            self.model = abs_util.IntendedTrajectoryPredictorTransformer(opts)
        else:
            self.model = abs_util.IntendedTrajectoryPredictor(opts)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)
        self.model = self.model.cuda(device=opts.gpu_id)
        self.total_iter = 0

    def init_dataset(self):
        self.dataloader = RandomWalks.data_loader(opts)

    def define_criterion(self):
        if self.opts.trans_align_loss:
            self.align_loss_fn = seq_alignment.TransitionAlignmentLoss()
        else:
            self.align_loss_fn = seq_alignment.AlignmentLoss()
            
        self.prior_loss_fn = seq_alignment.ConstantVelocityPrior()
        
        self.sf_loss_fn = abs_util.ScoreFunctionEstimator()
        self.sf_prior_loss_fn = abs_util.ScoreFunctionEstimator()
        if self.opts.variable_nseg:
            self.sf_latent_loss_fn = abs_util.ScoreFunctionEstimator()

    def set_input(self, batch):
        self.batch = batch
        self.trj_gt = batch.view(-1,1,2).float().cuda(device=opts.gpu_id)
        return

    def get_current_visuals(self):
        vis_dict = {}

        vis_dict['img_gt'] = RandomWalks.vis_walk(self.trj_gt[:,0,:].detach().cpu().numpy())
        vis_dict['img_pred'] = RandomWalks.vis_walk(self.trj_pred[:,0,:].detach().cpu().numpy())

        nsegs = len(self.model.primitives)
        for s in range(nsegs):
            img_prim_s = RandomWalks.vis_walk(self.model.primitives[s][0][:,0,:].detach().cpu().numpy())
            if s > 0:
                img_prims = np.hstack((img_prims, img_prim_s))
            else:
                img_prims = img_prim_s

        vis_dict['img_prims'] = img_prims
        return vis_dict

    def forward(self):
        # if self.opts.variable_nseg:
        #     self.trj_pred, probs, samples, z_probs, z_samples = self.model.forward(self.trj_gt)
        # else:
        #     self.trj_pred, probs, samples = self.model.forward(self.trj_gt)
        # z_probs and z_samples are set to None.
        self.trj_pred, probs, samples, z_probs, z_samples = self.model.forward(self.trj_gt)
            
        self.align_loss = self.align_loss_fn(self.trj_pred, self.trj_gt)/self.trj_gt.shape[0]
        self.sf_loss = self.sf_loss_fn.forward(self.align_loss, probs, samples)

        self.prior_loss = 0
        self.sf_prior_loss = 0

        for prim in self.model.primitives:
            l = self.prior_loss_fn(prim[0][:,0,:])
            self.prior_loss += l
            self.sf_prior_loss += self.sf_prior_loss_fn.forward(l, prim[1], prim[2])

        self.total_loss = 0

        self.total_loss += self.opts.align_loss_wt*self.align_loss
        self.total_loss += self.opts.prior_loss_wt*self.prior_loss
        if self.opts.variable_nseg:
            ## z_sample = 0 -> continue, so we add a loss for every continue decision
            parsimony_loss = (1-z_samples.float())*self.opts.parsimony_loss_wt

            self.sf_latent_loss = self.sf_latent_loss_fn.forward(self.total_loss + parsimony_loss, z_probs, z_samples)
            self.total_loss += self.opts.sf_loss_wt*self.sf_latent_loss
            self.register_scalars({'sf_latent_loss': self.opts.sf_loss_wt*self.sf_latent_loss.item()})

        self.total_loss += self.opts.align_loss_wt*self.opts.sf_loss_wt*self.sf_loss
        self.total_loss += self.opts.prior_loss_wt**self.opts.sf_loss_wt*self.sf_prior_loss

        self.register_scalars({
            'loss': self.total_loss.item(),
            'align_loss': self.align_loss.item(),
            'prior_loss': self.opts.prior_loss_wt*self.prior_loss.item(),
            'sf_loss': self.sf_loss.item(),
            'sf_prior_loss': self.sf_prior_loss.item(),
        })
        self.total_iter += 1


def main(_):
    torch.manual_seed(0)
    trainer = PrimitiveDiscoveryTrainer(opts)
    trainer.init_training()
    trainer.train()
    
if __name__ == '__main__':
    app.run(main)