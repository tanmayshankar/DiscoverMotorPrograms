'''
Sample usage:
 both joints (nz 64):
    python -m sfd.Experiments.abstraction.mime_im2trj --plot_scalars --display_visuals=True --display_freq=5000 --st_space=joint -optim_bs=2 --gpu_id=0 --nz=64 --nh=64 --num_epochs=300 --align_loss_wt=0.1 --len_loss_wt=0.1 --lr_step_epoch_freq=50 --ds_freq=20 --name=mime_img2trj_baseline_debug
'''
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

from ...DataLoaders import MIME_Img_DataLoader as MIME
from ...nnutils import train_utils
from ...nnutils import seq_alignment
from ...nnutils import geometry as geom_util
from ...utils import baxter_vis
from . import abstraction_utils as abs_util

flags.DEFINE_enum('st_space', 'joint', ['ee_r', 'ee_l', 'ee_all', 'joint', 'joint_ra', 'joint_la', 'joint_both'], 'State space')
flags.DEFINE_integer('n_state', 0, 'Dimension of state space. Set automatically based on EE vs joint space')
flags.DEFINE_float('align_loss_norm', 2, 'L-p norm')
flags.DEFINE_float('align_loss_wt', 1, 'Weight for alignment loss')
flags.DEFINE_float('sf_loss_wt', 0.1, 'Weight of pseudo loss for SF estimator')
flags.DEFINE_boolean('normalize_loss',False,'Whether to normalize sequence loss.')
flags.DEFINE_float('len_loss_wt', 0, 'Weight of loss penalizing difference between GT and pred lengths')


class Img2TrjTrainer(train_utils.Trainer):
    def trajectory_img_seq(self, trj, pad_filler=False, pad_alpha=1):
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
        opts = self.opts
        return torch.norm(s2-s1, p=self.opts.align_loss_norm, dim=-1)

    def joints_img(self, s):
        '''
        Args:
            s: joint angles, as a numpy array
        Returns:
            img: image showing robot
        '''
        if self.opts.st_space == 'joint_ra':
            img_js = self.baxter_visualizer.set_joint_pose_return_image(s, arm='right')
        elif self.opts.st_space == 'joint_la':
            img_js = self.baxter_visualizer.set_joint_pose_return_image(s, arm='left')
        elif self.opts.st_space == 'joint_both':
            img_js = self.baxter_visualizer.set_joint_pose_return_image(s, arm='both')
        elif self.opts.st_space == 'joint':
            jd = self.dataloader.dataset.recreate_dictionary('full', s)
            js_vis = np.array([jd[k] for k in self.baxter_visualizer.environment.robot_joints])
            img_js = self.baxter_visualizer.set_joint_pose_return_image(js_vis, arm='both')
        return img_js

    def define_model(self):
        opts = self.opts
        self.baxter_visualizer = baxter_vis.MujocoVisualizer()
        self.model = abs_util.Imgs2TrajectoryPredictor(opts)
        self.model = self.model.cuda()
        
    def init_dataset(self):
        self.dataloader = MIME.data_loader(self.opts, split='train')

    def define_criterion(self):
        self.align_loss_fn = seq_alignment.AlignmentLoss(dist_fn=self.state_dist_fn)
        self.sf_loss_fn = abs_util.ScoreFunctionEstimator()

    def set_input(self, batch):
        opts = self.opts

        self.invalid_batch = 1-batch['is_valid'][0]

        if not(self.invalid_batch):
            self.batch = batch
            if opts.st_space == 'joint':
                self.trj_gt = batch['joint_angle_trajectory']

            self.trj_gt = self.trj_gt.view(-1, 1, self.opts.n_state).float().cuda()
            self.imgs = batch['imgs'][0].cuda()
        return

    def get_current_visuals(self):
        vis_dict = {}

        trj_pred = self.trj_pred[:,0,:].detach().cpu().numpy()
        vis_dict['trj_pred'] = self.trajectory_img_seq(trj_pred)

        trj_gt = self.trj_gt[:,0,:].detach().cpu().numpy()
        vis_dict['trj_gt'] = self.trajectory_img_seq(trj_gt)

        trj_pred_align = trj_pred[self.align_loss_fn._soln_inds[1], :]
        vis_dict['trj_pred_align'] = self.trajectory_img_seq(trj_pred_align)

        return vis_dict

    def forward(self):
        self.trj_pred, probs, samples = self.model.forward(self.imgs)
        if self.opts.normalize_loss:
            self.align_loss = self.align_loss_fn(self.trj_pred, self.trj_gt)/self.trj_gt.shape[0]
        else:
            self.align_loss = self.align_loss_fn(self.trj_pred, self.trj_gt)

        self.len_loss = abs(self.trj_gt.shape[0] - self.trj_pred.shape[0])*self.opts.len_loss_wt
        self.reinforce_loss = self.opts.align_loss_wt*self.align_loss+self.opts.len_loss_wt*self.len_loss

        self.sf_loss = self.sf_loss_fn.forward(self.reinforce_loss, probs, samples)

        self.total_loss = 0
        self.total_loss += self.opts.align_loss_wt*self.align_loss
        self.total_loss += self.opts.sf_loss_wt*self.sf_loss

        self.register_scalars({
            'loss': self.total_loss.item(),
            'align_loss': self.align_loss.item(),
            'len_loss': self.len_loss,
            'sf_loss': self.sf_loss.item(),

            'pred_len': self.trj_pred.shape[0],
            'gt_len': self.trj_gt.shape[0],
        })


def main(_):
    opts = flags.FLAGS
    if opts.st_space == 'ee_r' or opts.st_space == 'ee_l':
        opts.n_state = 7
    if opts.st_space == 'joint_ra' or opts.st_space == 'joint_la':
        opts.n_state = 7
    if opts.st_space == 'joint_both':
        opts.n_state = 14
    elif opts.st_space == 'ee_all':
        opts.n_state = 14
    elif opts.st_space == 'joint':
        opts.n_state = 17
    opts.logging_dir = osp.join(opts.logging_dir, 'mime_img')

    torch.manual_seed(0)
    trainer = Img2TrjTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)