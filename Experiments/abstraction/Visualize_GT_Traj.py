'''
Sample usage: 

On DevFair:

python3 -m SkillsfromDemonstrations.Experiments.abstraction.Visualize_GT_Traj --MIME_dir=/checkpoint/tanmayshankar/MIME/ --checkpoint_dir=saved_models/ --pretrain_skillnet_name=mime_plan_ba_fns_64

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
from PIL import Image

# flags.DEFINE_string('network_dir',None,'Path to parent folder of model that should be loaded.')
flags.DEFINE_string('base_save_visual_dir','/checkpoint/tanmayshankar/results/','Base path to save visuals')
flags.DEFINE_enum('task_todo','eval',['eval','retrieve'],'What to make the evaluator object do.')
flags.DEFINE_bool('merge_gifs',False,'Whether to merge gifs or not.')

class PrimitiveDiscoverEvaluator(PrimitiveDiscoveryTrainer):

	def setup_testing(self, split='val'):		
		self.print_freq = 100		
		self.split = split
		self.init_dataset(split=split)
		# self.define_model()
		# self.define_criterion()

		# # Save_dir goes to /saved_models - Join with the opts.name to evaluate particular run / model.
		# # self.network_dir = os.path.join(self.save_dir,self.opts.name)

		# if self.opts.num_pretrain_epochs > 0:
		# 	# To load model from a different run: 
		# 	if self.opts.network_dir: 
		# 		self.load_network(self.model, 'pred', 'latest', network_dir=self.opts.network_dir)
		# 	else:
		# 		self.load_network(self.model, 'pred', self.opts.num_pretrain_epochs)

	# Override so that we can control split.
	def init_dataset(self, split):
		self.dataloader = MIME.data_loader(self.opts, split=split)

	# # Define function to merge GIFs into single GIF. 
	# def merge_gifs(self, gif_list, save_filename, gif_size=400):
		
	# 	mpy_gifs = []
	# 	# Create MPY object of each GIF. 
	# 	mpy_gifs.append(mpy.VideoFileClip(gif_list[0]).resize(height=gif_size))
		
	# 	for i in range(1,len(gif_list)):
	# 		# Create MPY object of all GIFs.
	# 		mpy_gifs.append(mpy.VideoFileClip(gif_list[i]).resize(height=gif_size))
		
	# 	# Now merge them. 
	# 	# The shape of input array to clips array determines output array display shape. (For example, 2x2 --> 2x2 display.)
	# 	animation = mpy.clips_array([mpy_gifs])
		
	# 	# Save GIFs. 
	# 	animation.write_gif(save_filename, fps=20, verbose=False, logger=None)

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
		base_gif_list = ["GT"]
		gif_list = [base_name.format(suffix) for suffix in base_gif_list]
		# gif_list = [vis_dict[key] for key in ['trj_gt','trj_pred_align','trj_pred','primitives_pred']]
   
		if self.opts.merge_gifs:
			# Right now just passing to merge. 
			self.merge_gifs(gif_list,save_filename=base_name.format("Merge"))

	def save_gif(self, counter, gt_gif):
		# Create a save directory if it doesn't exist. 
		save_path = os.path.join(self.opts.base_save_visual_dir,"GroundTruthTestGifs")
		if not(os.path.isdir(save_path)):
			os.mkdir(save_path)
		save_path = os.path.join(save_path,self.split)
		if not(os.path.isdir(save_path)):
			os.mkdir(save_path)
		save_path = os.path.join(save_path,"Traj_{0}".format(counter))
		if not(os.path.isdir(save_path)):
			os.mkdir(save_path)

		# base_name = os.path.join(save_path,"{0}.gif")
		# imageio.mimsave(base_name.format("GT"),gt_gif.astype(np.uint8))	
		base_name = os.path.join(save_path,"Img_{0}.gif")
		for t in range(len(gt_gif)):
			img = Image.fromarray(gt_gif[t].astype('uint8'))
			img.save(base_name.format(t))

	def evaluate(self):
		
		# HARDCODE INDICES:
		indices = np.array([  8,   9,  16,  58,  78,  79,  86, 127, 129, 131, 154, 
			176, 187, 202, 206, 232, 260, 268, 270, 290, 291, 303, 321, 333, 351, 
			364, 366, 382, 385, 396, 427, 428, 429, 454, 458, 462, 513, 516, 517, 
			531, 545, 547, 591, 619, 624, 637, 646, 656, 680, 698, 705, 738, 741, 754, 762, 779, 798, 827, 832, 836])		

		# For every element in the dataset.
		for i, batch in enumerate(self.dataloader):

			if i in indices:
				print(i)
				self.set_input(batch)

				trj_gt = self.trj_gt[:,0,:].detach().cpu().numpy()

				gt_gif = self.trajectory_img_seq(trj_gt)
	
				self.save_gif(i, gt_gif)		

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
	# Create evaluator instance. 
	evaluator = PrimitiveDiscoverEvaluator(opts)

	# Set up testing. 
	evaluator.setup_testing(split='test')

	if opts.task_todo=='eval':
		evaluator.evaluate()
	

if __name__ == '__main__':
	app.run(main)