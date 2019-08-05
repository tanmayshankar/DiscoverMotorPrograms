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

from ...DataLoaders import MIME_DataLoader as MIME
from ...nnutils import test_utils
from ...nnutils import seq_alignment
from ...nnutils import geometry as geom_util
from ...utils import baxter_vis
from . import abstraction_utils as abs_util


flags.DEFINE_enum('st_space', 'ee_r', ['ee_r', 'ee_l', 'ee_all', 'joint', 'joint_ra', 'joint_la', 'joint_both'], 'State space')
flags.DEFINE_integer('n_state', 0, 'Dimension of state space. Set automatically based on EE vs joint space')
flags.DEFINE_float('align_loss_norm', 2, 'L-p norm')

class PrimitiveDiscoveryTester(test_utils.Tester):

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
                except e:
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
        if 'joint' in opts.st_space:
            return torch.norm(s2-s1, p=self.opts.align_loss_norm, dim=-1)

    def state_post_process_fn(self, s):
        return s

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

        # Now creating model either with the transformer or recurrent encoder decoder. 
        self.model = abs_util.IntendedTrajectoryPredictor(opts)
        self.load_network(self.model, 'pred', opts.num_train_epoch)
        self.model = self.model.cuda()

    def init_dataset(self):
        self.dataloader = MIME.data_loader(self.opts, split='val')

    def set_input(self, batch):
        opts = self.opts
        self.batch = batch
        if opts.st_space == 'ee_l':
            self.trj_gt = batch['left_trajectory']
        elif opts.st_space == 'ee_r':
            self.trj_gt = batch['right_trajectory']
        elif opts.st_space == 'ee_all':
            self.trj_gt = torch.cat((batch['left_trajectory'], batch['right_trajectory']), dim=-1)
        elif opts.st_space == 'joint':
            self.trj_gt = batch['joint_angle_trajectory']
        elif opts.st_space == 'joint_ra':
            self.trj_gt = batch['ra_trajectory']
        elif opts.st_space == 'joint_la':
            self.trj_gt = batch['la_trajectory']
        elif opts.st_space == 'joint_both':
            # according to baxter_vis, the plans were left, right joints
            self.trj_gt = torch.cat((batch['la_trajectory'], batch['ra_trajectory']), dim=-1)

        self.trj_gt = self.trj_gt.view(-1, 1, self.opts.n_state).float().cuda(device=opts.gpu_id)
        return
