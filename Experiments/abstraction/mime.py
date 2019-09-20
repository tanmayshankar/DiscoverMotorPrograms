""""
Script for training a simple inverse model using a regression loss.
Sample usage:
      python -m sfd.Experiments.abstraction.mime --plot_scalars --display_visuals=True --display_freq=100 --name=mime_joint_tal --st_space=joint --MIME_dir=/checkpoint/shubhtuls/data/MIME/ --num_epochs=300 --lr_step_epoch_freq=50 --ds_freq=50 --optim_bs=10 --trans_align_loss

With pretrained primitive:
      python -m sfd.Experiments.abstraction.mime --plot_scalars --display_visuals=True --display_freq=2000 --st_space=joint_ra -optim_bs=2 --variable_ns=False --gpu_id=0 --pretrain_skillnet_name=mime_plan_ra_fns --num_pretrain_skillnet_epoch=80 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/shubhtuls/data/MIME/ --num_epochs=300 --lr_step_epoch_freq=50 --ds_freq=20 --name=mime_ra_fns_ptdec

With pretrained primitive + fixed nseg:
      python -m sfd.Ex periments.abstraction.mime --plot_scalars --display_visuals=True --display_freq=2000 --name=mime_ra_fns_nseg6_ptdec --st_space=joint_ra -optim_bs=2 --variable_ns=False --variable_nseg=False --n_skill_segments=6 --gpu_id=0 --pretrain_skillnet_name=mime_plan_ra_fns --num_pretrain_skillnet_epoch=80 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/shubhtuls/data/MIME/ --num_epochs=300 --lr_step_epoch_freq=50 --ds_freq=20

    python -m sfd.Experiments.abstraction.mime --plot_scalars --display_visuals=True --display_freq=5000 --st_space=joint_ra -optim_bs=2 --variable_ns=False --gpu_id=0 --pretrain_skillnet_name=mime_plan_ra_fns --num_pretrain_skillnet_epoch=80 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/shubhtuls/data/MIME/ --num_epochs=300 --lr_step_epoch_freq=50 --ds_freq=20 --name=mime_ra_fns_ptdec_vnsegb0_L1_subs --subsample_trj --lpred_p_bias=0 --align_loss_norm=1

    both joints:
    python -m sfd.Experiments.abstraction.mime --plot_scalars --display_visuals=True --display_freq=5000 --st_space=joint_both -optim_bs=2 --variable_ns=False --gpu_id=0 --pretrain_skillnet_name=mime_plan_ba_fns --num_pretrain_skillnet_epoch=100 --fixed_skillnet --nz=32 --nh=32 --MIME_dir=/checkpoint/shubhtuls/data/MIME/ --num_epochs=300 --lr_step_epoch_freq=50 --ds_freq=20 --name=mime_ba_fns_ptdec_vnsegb0_ll0pt01_sl0pt1_L2 --subsample_trj=False --lpred_p_bias=0 --align_loss_norm=2 --len_loss_wt=0.01 --step_loss_wt=0.1

    both joints (ft):
    python -m sfd.Experiments.abstraction.mime --plot_scalars --display_visuals=True --display_freq=5000 --st_space=joint_both -optim_bs=2 --variable_ns=False --gpu_id=0 --pretrain_skillnet_name=mime_plan_ba_fns --num_pretrain_epochs=100 --fixed_skillnet=False --nz=32 --nh=32 --MIME_dir=/checkpoint/shubhtuls/data/MIME/ --num_epochs=300 --lr_step_epoch_freq=50 --ds_freq=20 --name=mime_ba_fns_ftdec_vnsegb0_L2 --subsample_trj=False --lpred_p_bias=0 --align_loss_norm=2

    both joints (nz 64):
    python -m sfd.Experiments.abstraction.mime --plot_scalars --display_visuals=True --display_freq=5000 --st_space=joint_both -optim_bs=2 --variable_ns=False --gpu_id=0 --pretrain_skillnet_name=mime_plan_ba_fns_64 --num_pretrain_skillnet_epoch=100 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/shubhtuls/data/MIME/ --num_epochs=300 --lr_step_epoch_freq=50 --ds_freq=20 --name=mime_ba_fns_ptdec_vnsegb0_nzh64_ll0pt01_sl1_L2 --subsample_trj=False --lpred_p_bias=0 --align_loss_norm=2 --len_loss_wt=0.01 --step_loss_wt=1

python ~/scripts/cluster_run.py --setup=/private/home/shubhtuls/scripts/init_mujoco_cluster.sh --cmd=''
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from absl import app
from absl import flags
import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torchvision

from ...DataLoaders import MIME_DataLoader as MIME
from ...nnutils import train_utils
from ...nnutils import seq_alignment
from ...nnutils import geometry as geom_util
from ...utils import baxter_vis
from . import abstraction_utils as abs_util
from IPython import embed
import cProfile
from memory_profiler import profile
   
flags.DEFINE_boolean('transformer', True, 'Whether or not to use the transformer model (or default recurrent encoder decoder).')
flags.DEFINE_enum('st_space', 'joint_both_gripper', ['ee_r', 'ee_l', 'ee_all', 'joint', 'joint_ra', 'joint_la', 'joint_both', 'joint_both_gripper'], 'State space')
flags.DEFINE_integer('n_state', 0, 'Dimension of state space. Set automatically based on EE vs joint space')
flags.DEFINE_float('align_loss_wt', 1, 'Weight for alignment loss')
flags.DEFINE_float('align_loss_norm', 2, 'L-p norm')
flags.DEFINE_float('len_loss_wt', 0, 'Weight of loss penalizing difference between GT and pred lengths')
flags.DEFINE_float('step_loss_wt', 0, 'Weight for step hinge loss')
flags.DEFINE_float('step_loss_hinge', 0.5, 'Hinge loss threshold')
flags.DEFINE_boolean('trans_align_loss', False, 'Use transition alignment loss instead of basic alignment loss')
flags.DEFINE_boolean('fixed_skillnet', False, 'Keep the params of skillnet fixed')
flags.DEFINE_boolean('subsample_trj', False, 'Subsample half-full trajectory')
flags.DEFINE_boolean('vae_enc', False, 'Whether to use VAE encoder transformer or not.')
flags.DEFINE_float('sf_loss_wt', 0.1, 'Weight of pseudo loss for SF estimator')
flags.DEFINE_float('prior_loss_wt', 0, 'Weight for prior loss on each subsequence')
flags.DEFINE_float('kld_loss_wt', 0, 'Weight for KL Divergence loss if using VAE encoder.')
flags.DEFINE_float('parsimony_loss_wt', 0.01, 'Additional cost for using each primitive')
flags.DEFINE_integer('num_pretrain_skillnet_epoch', 0, 'Use a pretrained skill decoder if > 0')
flags.DEFINE_string('pretrain_skillnet_name', 'mime_plan_ba_fns', 'Name of pretrained net')
flags.DEFINE_boolean('normalize_loss',False,'Whether to normalize sequence loss.')
flags.DEFINE_string('network_dir',None,'Directory to load network from.')
flags.DEFINE_boolean('shuffle',True,'Whether to shuffle dataset or not.')
flags.DEFINE_boolean('profile',False,'Whether to profile.')

class PrimitiveDiscoveryTrainer(train_utils.Trainer):

    def ee_dist_fn(self, s1, s2, trans_wt=1, rot_wt=0.2):
        t1, q1 = torch.split(s1, [3, 4], dim=-1)
        t2, q2 = torch.split(s2, [3, 4], dim=-1)
        dt = torch.norm(t2-t1, p=2, dim=-1)
        dq = geom_util.quat_dist(q1, q2)
        return dt*trans_wt + dq*rot_wt

    # @profile
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

        image_stack = np.stack(imgs, axis=0)
        del imgs
        return image_stack

    def state_dist_fn(self, s1, s2):
        opts = self.opts
        if opts.st_space == 'ee_r' or opts.st_space == 'ee_l':
            return self.ee_dist_fn(s1, s2)
        elif opts.st_space == 'ee_all':
            ee1_r, ee1_l = torch.split(s1, [7, 7], dim=-1)
            ee2_r, ee2_l = torch.split(s1, [7, 7], dim=-1)
            return self.ee_dist_fn(ee1_r, ee2_r) + self.ee_dist_fn(ee1_l, ee2_l)
        elif 'joint' in opts.st_space:
            return torch.norm(s2-s1, p=self.opts.align_loss_norm, dim=-1)

    def ee_post_process_fn(self, s):
        t, q = torch.split(s, [3, 4], dim=-1)
        q = q/torch.norm(q, dim=-1, keepdim=True)
        return torch.cat([t, q], dim=-1)

    def state_post_process_fn(self, s):
        opts = self.opts
        if opts.st_space == 'ee_l' or opts.st_space == 'ee_r':
            return self.ee_post_process_fn(s)
        elif opts.st_space == 'ee_all':
            ee_r, ee_l = torch.split(s, [7, 7], dim=-1)
            return torch.cat([self.ee_post_process_fn(ee_r), self.ee_post_process_fn(ee_l)], dim=-1)
        else:
            return s
    
    # @profile
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
        elif self.opts.st_space == 'joint_both_gripper':
            img_js = self.baxter_visualizer.set_joint_pose_return_image(s, arm='both', gripper=True)
        elif self.opts.st_space == 'joint':
            jd = self.dataloader.dataset.recreate_dictionary('full', s)
            js_vis = np.array([jd[k] for k in self.baxter_visualizer.environment.robot_joints])
            img_js = self.baxter_visualizer.set_joint_pose_return_image(js_vis, arm='both')
        return img_js

    def define_model(self):
        opts = self.opts
        self.baxter_visualizer = baxter_vis.MujocoVisualizer()

        # Now creating model either with the transformer or recurrent encoder decoder. 
        if self.opts.transformer:
            if self.opts.vae_enc:
                self.model = abs_util.IntendedTrajectoryPredictorTransformerVAE(opts)
            else:
                self.model = abs_util.IntendedTrajectoryPredictorTransformer(opts)
        else:
            self.model = abs_util.IntendedTrajectoryPredictor(opts)

        # if opts.num_pretrain_skillnet_epoch > 0:
        #     self.load_network(
        #         self.model.primitive_decoder, 'pred', opts.num_pretrain_skillnet_epoch,
        #         network_dir=osp.join(opts.checkpoint_dir, opts.pretrain_skillnet_name),
        #         module_name='decoder'
        #     )

        if opts.num_pretrain_skillnet_epoch > 0:
            self.load_network(
                self.model.primitive_decoder, 'pred', 'latest',
                network_dir=osp.join(opts.checkpoint_dir, opts.pretrain_skillnet_name),
                module_name='decoder'
            )

        if opts.num_pretrain_epochs > 0:
            # To load model from a different run: 
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs, network_dir=opts.network_dir)
        if opts.num_pretrain_epochs==-1:
            self.load_network(self.model, 'pred', 'latest', network_dir=opts.network_dir)

        # embed()
        if opts.fixed_skillnet:
            for param in self.model.primitive_decoder.parameters():
                param.requires_grad = False
        self.model = self.model.cuda(device=opts.gpu_id)
        self.total_iter = 0

    def init_dataset(self):
        self.dataloader = MIME.data_loader(self.opts, split='train', shuffle=self.opts.shuffle)

    # @profile
    def define_criterion(self):
        if self.opts.trans_align_loss:
            # TODO: make sure this works for other spaces
            self.align_loss_fn = seq_alignment.TransitionAlignmentLoss(dist_fn=self.state_dist_fn)
        else:
            self.align_loss_fn = seq_alignment.AlignmentLoss(dist_fn=self.state_dist_fn)

        self.step_loss_fn = seq_alignment.StepLoss(dist_fn=self.state_dist_fn, hinge=self.opts.step_loss_hinge)

        self.prior_loss_fn = seq_alignment.ConstantVelocityPrior()

        self.sf_loss_fn = abs_util.ScoreFunctionEstimator()
        self.sf_prior_loss_fn = abs_util.ScoreFunctionEstimator()
        if self.opts.variable_nseg:
            self.sf_latent_loss_fn = abs_util.ScoreFunctionEstimator()

    def sample_subtrajectory(self, trj):
        N = trj.shape[0]
        N_s = np.random.randint(N//3, N) + 1
        init = np.random.randint(N - N_s + 1)
        return trj[init:(init + N_s)]

    # @profile
    def set_input(self, batch):
        opts = self.opts

        self.invalid_batch = 1-batch['is_valid'][0]

        if not(self.invalid_batch):
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
            elif opts.st_space == 'joint_both_gripper':
                # Feed to Baxter Viz in this order: Left arm, Right arm, Left Gripper, Right Gripper.            
                self.trj_gt = torch.cat((batch['la_trajectory'], batch['ra_trajectory'], batch['left_gripper'].unsqueeze(2), batch['right_gripper'].unsqueeze(2)), dim=-1)

            self.trj_gt = self.trj_gt.view(-1, 1, self.opts.n_state).float().cuda(device=opts.gpu_id)
            if self.opts.subsample_trj:
                self.trj_gt = self.sample_subtrajectory(self.trj_gt)

        return

    # @profile
    def get_current_visuals(self):
        vis_dict = {}

        trj_pred = self.trj_pred[:,0,:].detach().cpu().numpy()
        vis_dict['trj_pred'] = self.trajectory_img_seq(trj_pred)

        trj_gt = self.trj_gt[:,0,:].detach().cpu().numpy()
        vis_dict['trj_gt'] = self.trajectory_img_seq(trj_gt)

        trj_pred_align = trj_pred[self.align_loss_fn._soln_inds[1], :]
        vis_dict['trj_pred_align'] = self.trajectory_img_seq(trj_pred_align)

        nsegs = len(self.model.primitives)
        img_prims = []
        for s in range(nsegs):
            trj_prim_s = self.model.primitives[s][0][:,0,:].detach().cpu().numpy()
            img_prims.append(self.trajectory_img_seq(trj_prim_s, pad_alpha = (1 if s==(nsegs-1) else 0.8)))
        if (len(img_prims) > 0):
            vis_dict['primitives_pred'] = np.concatenate(img_prims, axis=0)

        return vis_dict

    # @profile
    def forward(self):
        # print("GROUND TRUTH TRAJECTORY: ", self.trj_gt)
        fake_fnseg_iterations = 50000

        t_a = time.time()
        if self.opts.variable_nseg:
            if self.total_iter>fake_fnseg_iterations:
                self.trj_pred, probs, samples, z_probs, z_samples = self.model.forward(self.trj_gt, fake_fnseg=False)
            else:
                self.trj_pred, probs, samples, z_probs, z_samples = self.model.forward(self.trj_gt, fake_fnseg=True)
        else:            
            self.trj_pred, probs, samples, z_probs, z_samples = self.model.forward(self.trj_gt)
        t_b = time.time()

        self.trj_pred = self.state_post_process_fn(self.trj_pred)
        # self.trj_pred.retain_grad()

        # self.align_loss = self.align_loss_fn(self.trj_pred[:,0,:], self.trj_gt[:,0,:])/self.trj_gt.shape[0]

        # if self.opts.divide==True:
        #     self.align_loss = self.align_loss_fn(self.trj_pred, self.trj_gt)/self.trj_gt.shape[0]
        # else:
        #     self.align_loss = self.align_loss_fn(self.trj_pred, self.trj_gt)

        # pdb.set_trace()

        t_c = time.time()
        if self.opts.normalize_loss:
            self.align_loss = self.align_loss_fn(self.trj_pred, self.trj_gt)/self.trj_gt.shape[0]
        else:
            self.align_loss = self.align_loss_fn(self.trj_pred, self.trj_gt)
        t_d = time.time()

        self.step_loss = self.step_loss_fn(self.trj_pred[:,0,:])*self.opts.step_loss_wt

        self.len_loss = abs(self.trj_gt.shape[0] - self.trj_pred.shape[0])*self.opts.len_loss_wt
        # self.align_loss += self.len_loss 
        self.reinforce_loss = self.opts.align_loss_wt*self.align_loss+self.opts.len_loss_wt*self.len_loss+ self.opts.step_loss_wt*self.step_loss

        t_e = time.time()
        if self.opts.variable_ns:
            self.sf_loss = self.sf_loss_fn.forward(self.reinforce_loss, probs, samples)
        else:
            self.sf_loss = self.align_loss*0

        self.prior_loss = self.align_loss*0
        self.sf_prior_loss = self.align_loss*0

        for prim in self.model.primitives:
            l = self.prior_loss_fn(prim[0][:,0,:])
            self.prior_loss += l
            if self.opts.variable_ns:
                self.sf_prior_loss += self.sf_prior_loss_fn.forward(l, prim[1], prim[2])

        t_f = time.time()

        # print("Time AB:",t_b-t_a)
        # print("Time BC:",t_c-t_b)
        # print("Time CD:",t_d-t_c)
        # print("Time DE:",t_e-t_d)
        # print("Time EF:",t_f-t_e)
        self.total_loss = 0

        self.total_loss += self.opts.align_loss_wt*self.align_loss
        self.total_loss += self.opts.prior_loss_wt*self.prior_loss
        self.total_loss += self.opts.len_loss_wt*self.len_loss
        self.total_loss += self.opts.step_loss_wt*self.step_loss

        if self.opts.variable_nseg:
            ## z_sample = 0 -> continue, so we add a loss for every continue decision
            self.parsimony_loss = ((1-z_samples.float())*self.opts.parsimony_loss_wt).sum()
            self.sf_latent_loss = self.sf_latent_loss_fn.forward(self.total_loss + self.parsimony_loss, z_probs, z_samples)

            # Prevent total collapse of NSeg to 2 or something. 
            # if self.total_iter>fake_fnseg_iterations:
            self.total_loss += self.opts.sf_loss_wt*self.sf_latent_loss
            self.register_scalars({'sf_latent_loss': self.opts.sf_loss_wt*self.sf_latent_loss.item(), 
                'parsimony_loss': self.parsimony_loss.item()})

        self.total_loss += self.opts.sf_loss_wt*self.sf_loss
        self.total_loss += self.opts.prior_loss_wt*self.opts.sf_loss_wt*self.sf_prior_loss

        if self.opts.vae_enc:
            self.total_loss += self.opts.kld_loss_wt*self.model.kld_loss
            self.register_scalars({'kld_loss': self.model.kld_loss})

        self.register_scalars({
            'loss': self.total_loss.item(),
            'align_loss': self.align_loss.item(),
            'len_loss': self.len_loss,
            'step_loss': self.step_loss.item(),
            'prior_loss': self.opts.prior_loss_wt*self.prior_loss.item(),
            'sf_loss': self.sf_loss.item(),
            'sf_prior_loss': self.sf_prior_loss.item(),
            'nz': len(self.model.primitives),
            'pred_len': self.trj_pred.shape[0],
            'gt_len': self.trj_gt.shape[0],
        })
        self.total_iter += 1    

# @profile
def main(_):
    opts = flags.FLAGS
    if opts.st_space == 'ee_r' or opts.st_space == 'ee_l':
        opts.n_state = 7
    if opts.st_space == 'joint_ra' or opts.st_space == 'joint_la':
        opts.n_state = 7
    if opts.st_space == 'joint_both':
        opts.n_state = 14
    if opts.st_space =='joint_both_gripper':
        opts.n_state = 16
    elif opts.st_space == 'ee_all':
        opts.n_state = 14
    elif opts.st_space == 'joint':
        opts.n_state = 17
    opts.logging_dir = osp.join(opts.logging_dir, 'mime')
    torch.manual_seed(0)
    trainer = PrimitiveDiscoveryTrainer(opts)
    trainer.init_training()

    # # Profiling to test VAE.
    # if opts.profile:
    #     cProfile.runctx('trainer.train()',globals(),locals())
    # else:
    #     trainer.train()

    # # Profiling to test VAE.
    if opts.profile:  
        pr = cProfile.Profile()
        pr.enable()
        trainer.train()
        pr.disable()

        print("################################")
        print("#### Print sorting by time. ####")
        print("################################")
        pr.print_stats(sort='time')

        print("################################")
        print("#### Print sorting tottime. ####")
        print("################################")
        pr.print_stats(sort='tottime')

        print("################################")
        print("#### Print sorting cumtime. ####")
        print("################################")
        pr.print_stats(sort='cumtime')
    else:
        trainer.train()

if __name__ == '__main__':
    app.run(main)