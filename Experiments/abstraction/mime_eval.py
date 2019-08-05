'''
Sample usage: 
On DevFair:
python3 -m SkillsfromDemonstrations.Experiments.abstraction.mime_eval --display_freq=5000 -optim_bs=2 --variable_ns=False --num_pretrain_skillnet_epoch=100 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/tanmayshankar/MIME/ --novariable_nseg --num_epochs=300 --lr_step_epoch_freq=50 --name=T119_fnseg_sl0pt5_al0pt1 --lpred_p_bias=0 --len_loss_wt=0.01 --step_loss_wt=0.5 --align_loss_wt=0.1 --transformer --nonormalize_loss --checkpoint_dir=saved_models/ --pretrain_skillnet_name=mime_plan_ba_fns_64 --st_space=joint_both
On Cluster / LearnFair:
python cluster_run.py --name=T119_fnseg_sl0pt5_al0pt1 --cmd='python3 -m SkillsfromDemonstrations.Experiments.abstraction.mime_eval --display_freq=5000 -optim_bs=2 --variable_ns=False --num_pretrain_skillnet_epoch=100 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/tanmayshankar/MIME/ --novariable_nseg --num_epochs=300 --lr_step_epoch_freq=50 --name=T119_fnseg_sl0pt5_al0pt1 --lpred_p_bias=0 --len_loss_wt=0.01 --step_loss_wt=0.5 --align_loss_wt=0.1 --transformer --nonormalize_loss --checkpoint_dir=saved_models/ --pretrain_skillnet_name=mime_plan_ba_fns_64'
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from absl import app, flags

import os.path as osp, os
import numpy as np, imageio
import torch, torch.nn as nn, torchvision

from ...DataLoaders import MIME_DataLoader as MIME
from ...nnutils import train_utils
from ...nnutils import seq_alignment
from ...nnutils import geometry as geom_util
from ...utils import baxter_vis
from . import abstraction_utils as abs_util
from .mime import PrimitiveDiscoveryTrainer
from IPython import embed
import IPython
from IPython.display import display, Image
import moviepy.editor as mpy

# flags.DEFINE_string('network_dir',None,'Path to parent folder of model that should be loaded.')
flags.DEFINE_string('base_save_visual_dir','/checkpoint/tanmayshankar/results/','Base path to save visuals')
flags.DEFINE_enum('task_todo','eval',['eval','retrieve'],'What to make the evaluator object do.')
flags.DEFINE_bool('merge_gifs',False,'Whether to merge gifs or not.')

class PrimitiveDiscoverEvaluator(PrimitiveDiscoveryTrainer):

	def setup_testing(self, split='val'):		
		self.print_freq = 100		
		self.split = split
		self.init_dataset(split=split)
		self.define_model()
		self.define_criterion()

		# Save_dir goes to /saved_models - Join with the opts.name to evaluate particular run / model.
		# self.network_dir = os.path.join(self.save_dir,self.opts.name)
		print("HELLO")
		if self.opts.num_pretrain_epochs > 0:
			# To load model from a different run: 
			if self.opts.network_dir: 
				self.load_network(self.model, 'pred', 'latest', network_dir=self.opts.network_dir)
			else:
				self.load_network(self.model, 'pred', self.opts.num_pretrain_epochs)

	# Override so that we can control split.
	def init_dataset(self, split):
		torch.manual_seed(0)
		self.dataloader = MIME.data_loader(self.opts, split=split)

	# Define function to merge GIFs into single GIF. 
	def merge_gifs(self, gif_list, save_filename, gif_size=400):
		
		mpy_gifs = []
		# Create MPY object of each GIF. 
		mpy_gifs.append(mpy.VideoFileClip(gif_list[0]).resize(height=gif_size))
		
		for i in range(1,len(gif_list)):
			# Create MPY object of all GIFs.
			mpy_gifs.append(mpy.VideoFileClip(gif_list[i]).resize(height=gif_size))
		
		# Now merge them. 
		# The shape of input array to clips array determines output array display shape. (For example, 2x2 --> 2x2 display.)
		animation = mpy.clips_array([mpy_gifs])
		
		# Save GIFs. 
		animation.write_gif(save_filename, fps=20, verbose=False, logger=None)

	def save_model_eval(self, counter, vis_dict):

		# Create a save directory if it doesn't exist. 
		save_path = os.path.join(self.opts.base_save_visual_dir,self.opts.name)		
		if not(os.path.isdir(save_path)):
			os.mkdir(save_path)
		save_path = os.path.join(save_path,self.split)
		if not(os.path.isdir(save_path)):
			os.mkdir(save_path)
		save_path = os.path.join(save_path,"Traj_{0}".format(counter))
		if not(os.path.isdir(save_path)):
			os.mkdir(save_path)

		# Save visuals and statistics etc. 		
		base_name = os.path.join(save_path,"{0}.gif")
		imageio.mimsave(base_name.format("Pred"),vis_dict['trj_pred'].astype(np.uint8))
		imageio.mimsave(base_name.format("GT"),vis_dict['trj_gt'].astype(np.uint8))
		imageio.mimsave(base_name.format("Pred_Align"),vis_dict['trj_pred_align'].astype(np.uint8))
		imageio.mimsave(base_name.format("Primitive_Pred"),vis_dict['primitives_pred'].astype(np.uint8))

		# Create merged GIF. 
		base_gif_list = ["GT","Pred_Align","Pred","Primitive_Pred"]
		gif_list = [base_name.format(suffix) for suffix in base_gif_list]
		# gif_list = [vis_dict[key] for key in ['trj_gt','trj_pred_align','trj_pred','primitives_pred']]
   
		if self.opts.merge_gifs:
			# Right now just passing to merge. 
			self.merge_gifs(gif_list,save_filename=base_name.format("Merge"))
	
		scalar_dict = { 'loss': self.total_loss.item(),
						'align_loss': self.align_loss.item(),
						'weighted_align_loss': (self.opts.align_loss_wt*self.align_loss).item(),
						'len_loss': self.len_loss,
						'step_loss': self.step_loss.item(),
						'prior_loss': self.opts.prior_loss_wt*self.prior_loss.item(),
						'sf_loss': self.sf_loss.item(),
						'sf_prior_loss': self.sf_prior_loss.item(),
						'nz': len(self.model.primitives),
						'pred_len': self.trj_pred.shape[0],
						'gt_len': self.trj_gt.shape[0]
					   }

		scalar_name = os.path.join(save_path,"Traj_{0}_Scalars.npy".format(counter))
		np.save(scalar_name, scalar_dict)

	def evaluate(self):
		
		# For every element in the dataset.
		for i, batch in enumerate(self.dataloader):

			if i%self.print_freq==0:
				print("Saving trajectory ",i)

			# Use the Trainer's set_input function to set self.trj_gt to the current datapoint. 
			self.set_input(batch)

			# Now run Forward from Trainer. 
			self.forward()

			# Now get current visuals.
			vis_dict = self.get_current_visuals()			

			self.save_model_eval(i, vis_dict)

	def retrieve_predictions(self):
		# Retrieve predictions from Trainer object - Predicted Primitives and Latent Zs. 
		latent_z_seq = self.model.latent_z_seq.squeeze(1).detach().cpu().numpy()
		primitives = [x[0].squeeze(1).detach().cpu().numpy() for x in self.model.primitives]
		return latent_z_seq, primitives

	def retrieve_skills(self):

		# Initialize lists of Zs and trajectories. 
		traj_list = []
		z_list = []		

		# For every trajectory in dataset: 
		for i, batch in enumerate(self.dataloader):

			if i%self.print_freq==0:
				print("Retrieving skills from Trajectory ",i)

			# Like evaluate function, set input then run forward.
			self.set_input(batch)
			self.forward()

			# Set numpy objects. 
			latent_z_seq, primitives = self.retrieve_predictions()

			# Append to overall list.
			traj_list.append(primitives)
			z_list.append(latent_z_seq)

		traj_array = np.array(traj_list)
		z_array = np.array(z_list)

		return traj_array, z_array

	def retrieve_and_cluster(self):
		trajectories, latent_zs = self.retrieve_skills()

		

def main(_):
	opts = flags.FLAGS
	if opts.st_space == 'ee_r' or opts.st_space == 'ee_l':
		opts.n_state = 7
	if opts.st_space == 'joint_ra' or opts.st_space == 'joint_la':
		opts.n_state = 7
	if opts.st_space == 'joint_both':
		opts.n_state = 14
	if opts.st_space == 'joint_both_gripper':
		opts.n_state = 16
	elif opts.st_space == 'ee_all':
		opts.n_state = 14
	elif opts.st_space == 'joint':
		opts.n_state = 17
	opts.logging_dir = osp.join(opts.logging_dir, 'mime')

	torch.manual_seed(0)  
	# Create evaluator instance. 
	evaluator = PrimitiveDiscoverEvaluator(opts)

	# Set up testing. 
	evaluator.setup_testing(split='test')

	if opts.task_todo=='eval':
		evaluator.evaluate()
	if opts.task_todo=='retrieve':
		evaluator.retrieve_and_cluster()
	

if __name__ == '__main__':
	app.run(main)