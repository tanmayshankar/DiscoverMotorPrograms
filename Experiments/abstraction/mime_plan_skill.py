"""
Script for training skill net.
Sample usage:
      python -m sfd.Experiments.abstraction.mime_plan_skill --plot_scalars --display_visuals=True --display_freq=5000 --name=mime_plan_ba_vns --st_space=joint_both --MIME_dir=/checkpoint/tanmayshankar/MIME/ --num_epochs=100 --lr_step_epoch_freq=50 --optim_bs=2 --variable_ns=True --gpu_id=0 --sf_loss_wt=0.1 --align_loss_wt=1.0 --num_pretrain_epochs=12
    with prob loss:
      python -m sfd.Experiments.abstraction.mime_plan_skill --plot_scalars --display_visuals=True --display_freq=5000 --name=mime_plan_ba_vns_pal --st_space=joint_both --MIME_dir=/checkpoint/tanmayshankar/MIME/ --num_epochs=100 --lr_step_epoch_freq=50 --optim_bs=2 --variable_ns=True --gpu_id=0 --sf_loss_wt=0.1 --align_loss_wt=1.0 --prob_align_loss --num_pretrain_epochs=6
    with fixed prim_len:
        python -m sfd.Experiments.abstraction.mime_plan_skill --plot_scalars --display_visuals=True --display_freq=5000 --name=mime_plan_bag_fns_nzh64 --st_space=joint_both --MIME_dir=/checkpoint/tanmayshankar/MIME/ --num_epochs=100 --lr_step_epoch_freq=50 --optim_bs=2 --variable_ns=False --gpu_id=0 --align_loss_wt=1.0 --nz=64 --nh=64 --pred_gripper=True --num_pretrain_epochs=8

with fixed prim_len + vae:
        python -m sfd.Experiments.abstraction.mime_plan_skill --plot_scalars --display_visuals=True --display_freq=5000 --name=mime_plan_bag_fns_nzh64_vae --st_space=joint_both --MIME_dir=/checkpoint/tanmayshankar/MIME/ --num_epochs=100 --lr_step_epoch_freq=50 --optim_bs=2 --variable_ns=False --gpu_id=0 --align_loss_wt=1.0 --nz=64 --nh=64 --pred_gripper=True --num_pretrain_epochs=8 --vae_enc

# python ~/scripts/cluster_run.py --setup=/private/home/shubhtuls/scripts/init_mujoco_cluster.sh --cmd=''
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

from ...DataLoaders import Plan_DataLoader as MIMEPlans
from ...nnutils import train_utils
from ...nnutils import seq_alignment
from ...nnutils import probabilistic_seq_alignment
from ...nnutils import geometry as geom_util
from ...utils import baxter_vis
from . import abstraction_utils as abs_util

flags.DEFINE_enum('st_space', 'joint_ra', ['joint_all','joint_ra', 'joint_la','joint_both'], 'State space')
flags.DEFINE_boolean('pred_gripper', False, 'Whether to encode and predict gripper as well')
flags.DEFINE_integer('n_state', 0, 'Dimension of state space. Set automatically based on state space')
flags.DEFINE_float('align_loss_wt', 1, 'Weight for alignment loss')
flags.DEFINE_boolean('prob_align_loss', False, 'Whether to use probabilistic align loss')
flags.DEFINE_float('sf_loss_wt', 0.1, 'Weight of pseudo loss for SF estimator')
flags.DEFINE_boolean('vae_enc', False, 'Whether the skill net encoder is a variational one')
flags.DEFINE_float('kld_loss_wt', 1, 'Weight for alignment loss')


class SkillTrainer(train_utils.Trainer):
    def trajectory_img_seq(self, trj, pad_filler=True, pad_alpha=1):
        '''
        Args:
            trj: T X nS numpy array
        Returns:
            imgs: T X H X W X 3
        '''
        opts = self.opts
        imgs = []
        for t, s_t in enumerate(trj):
            if 'joint' in opts.st_space:
                try:
                    img_t = self.joints_img(s_t)
                except:
                    img_t = np.zeros((600,600,3))
                    self.baxter_visualizer = baxter_vis.MujocoVisualizer()
            else:
                img_t = np.zeros((600,600,3))
            imgs.append(img_t)
        if pad_filler:
            for t_pad in range(3):
                imgs.append(img_t*(1-pad_alpha) + 255*pad_alpha)

        return np.stack(imgs, axis=0)


    def state_dist_fn(self, s1, s2):
        return torch.norm(s2-s1, p=2, dim=-1)

    def joints_img(self, s):
        '''
        Args:
            s: joint angles, as a numpy array
                7 or 14 or 17D based on st_space
        Returns:
            img: image showing robot
        '''
        if self.opts.st_space == 'joint_ra':
            img_js = self.baxter_visualizer.set_joint_pose_return_image(s, arm='right', gripper=self.opts.pred_gripper)
        elif self.opts.st_space == 'joint_la':
            img_js = self.baxter_visualizer.set_joint_pose_return_image(s, arm='left', gripper=self.opts.pred_gripper)
        elif self.opts.st_space == 'joint_both':
            img_js = self.baxter_visualizer.set_joint_pose_return_image(s, arm='both', gripper=self.opts.pred_gripper)
        else:
            img_js = None
        return img_js

    def define_model(self):
        opts = self.opts
        self._num_tot_iter = 0
        self.baxter_visualizer = baxter_vis.MujocoVisualizer()

        self.model = torch.nn.Module()
        if opts.vae_enc:
            self.model.encoder = abs_util.TrajectoryEncoderVAE(opts)
        else:
            self.model.encoder = abs_util.TrajectoryEncoder(opts)

        if opts.variable_ns:
            self.model.decoder = abs_util.PrimitiveDecoder(opts, max_prim_len=20)
        else:
            self.model.decoder = abs_util.PrimitiveDecoderKnownLength(opts)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)
        self.model = self.model.cuda(device=opts.gpu_id)
        self.total_iter = 0

    def init_dataset(self):
        self.dataloader = MIMEPlans.data_loader(self.opts, split='train')

    def define_criterion(self):
        self.align_loss_fn = seq_alignment.AlignmentLoss(dist_fn=self.state_dist_fn)
        self.prob_align_loss_fn = probabilistic_seq_alignment.ProbabilisticAlignmentLoss(dist_fn=self.state_dist_fn)
        self.sf_loss_fn = abs_util.ScoreFunctionEstimator()

    def set_input(self, batch):
        opts = self.opts
        self.invalid_batch = (batch['JA_Plan'].numel() < 1)
        if not self.invalid_batch:
            self.batch = batch
            self.trj_gt = batch['JA_Plan']
            if opts.pred_gripper:
                self.trj_gt = torch.cat((self.trj_gt, batch['Grip_Plan']), dim=-1)

            self.trj_gt = self.trj_gt.view(-1, 1, self.opts.n_state).float().cuda(device=opts.gpu_id)

        return

    def get_current_visuals(self):
        vis_dict = {}

        trj_pred = self.trj_pred[:,0,:].detach().cpu().numpy()
        vis_dict['trj_pred'] = self.trajectory_img_seq(trj_pred)

        trj_gt = self.trj_gt[:,0,:].detach().cpu().numpy()
        vis_dict['trj_gt'] = self.trajectory_img_seq(trj_gt)

        return vis_dict

    def forward(self):
        self._num_tot_iter +=1
        self.skill_latent, _ = self.model.encoder.forward(self.trj_gt)
        self.skill_latent = self.skill_latent[-1]

        if self.opts.vae_enc:
            self.skill_latent_mu, self.log_sigma = torch.chunk(self.skill_latent, 2, dim=1)
            std = torch.exp(0.5*self.log_sigma)
            eps = torch.randn_like(std)
            self.skill_latent = self.skill_latent_mu + eps*std
            self.kld_loss = -0.5 * torch.sum(1 + self.log_sigma - self.skill_latent_mu.pow(2) - self.log_sigma.exp())
            # if self._num_tot_iter >= 1000:
            # pdb.set_trace()
            
        if self.opts.variable_ns and self.opts.prob_align_loss:
            self.trj_pred_full, probs, samples = self.model.decoder.forward(self.skill_latent, max_unroll=True)
        else:
            self.trj_pred, probs, samples = self.model.decoder.forward(self.skill_latent)

        if self.opts.variable_ns and self.opts.prob_align_loss:
            q = probabilistic_seq_alignment.construct_transition_probs(
                probs[:,0,0], [0], [self.trj_pred_full.shape[0]])

            self.align_loss = self.prob_align_loss_fn.forward(
                self.trj_gt, self.trj_pred_full, q)/self.trj_gt.shape[0]
            self.trj_pred, _, _ = self.prob_align_loss_fn.sample(
                self.trj_pred_full.detach(), q.detach())
        else:
            self.align_loss = self.align_loss_fn(self.trj_pred, self.trj_gt)/self.trj_gt.shape[0]

        if self.opts.variable_ns and not self.opts.prob_align_loss:
            self.sf_loss = self.sf_loss_fn.forward(self.align_loss, probs, samples)
        else:
            self.sf_loss = self.align_loss*0

        self.total_loss = 0

        self.total_loss += self.opts.align_loss_wt*self.align_loss
        self.total_loss += self.opts.align_loss_wt*self.opts.sf_loss_wt*self.sf_loss

        if self.opts.vae_enc:
            self.total_loss += self.opts.kld_loss_wt*self.kld_loss
            self.register_scalars({'kld_loss': self.kld_loss.item()})

        self.register_scalars({
            'loss': self.total_loss.item(),
            'align_loss': self.align_loss.item(),
            'sf_loss': self.sf_loss.item(),

            'pred_len': self.trj_pred.shape[0],
            'gt_len': self.trj_gt.shape[0],
        })
        self.total_iter += 1


def main(_):
    opts = flags.FLAGS
    if opts.st_space == 'joint_ra' or opts.st_space == 'joint_ra':
        opts.n_state = 7
        if opts.st_space == 'joint_ra':
            opts.arm = 'right'
        else:
            opts.arm = 'left'
        if opts.pred_gripper:
            opts.n_state += 1

    elif opts.st_space == 'joint_both':
        opts.arm = 'both'
        opts.n_state = 14
        if opts.pred_gripper:
            opts.n_state += 2

    elif opts.st_space == 'joint_all':
        pass

    opts.logging_dir = osp.join(opts.logging_dir, 'mime_plans')

    torch.manual_seed(0)
    trainer = SkillTrainer(opts)
    trainer.init_training()
    trainer.train()


if __name__ == '__main__':
    app.run(main)