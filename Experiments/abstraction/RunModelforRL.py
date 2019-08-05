'''
Sample usage: 

python3 -m SkillsfromDemonstrations.Experiments.abstraction.RunModelforRL --num_pretrain_skillnet_epoch=100 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/tanmayshankar/MIME/ --novariable_nseg --transformer --checkpoint_dir=saved_models/ --pretrain_skillnet_name=mime_plan_ba_fns_64 --num_pretrain_epochs=100 --network_dir=saved_models/T313_fnseg_sl2pt0_grip_finetune --st_space=joint_both_gripper --variable_ns=False

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

class PrimitiveDiscoverEvaluator(PrimitiveDiscoveryTrainer):

	def setup_testing(self, split='val'):		
		self.print_freq = 100		
		self.split = split
		self.init_dataset(split=split)
		self.define_model()
		self.define_criterion()

		# Save_dir goes to /saved_models - Join with the opts.name to evaluate particular run / model.
		# self.network_dir = os.path.join(self.save_dir,self.opts.name)

		if self.opts.num_pretrain_epochs > 0:
			# To load model from a different run: 
			if self.opts.network_dir: 
				self.load_network(self.model, 'pred', 'latest', network_dir=self.opts.network_dir)
			else:
				self.load_network(self.model, 'pred', self.opts.num_pretrain_epochs)

		# Create skill net reference. 
		self.skill_network = self.model.primitive_decoder

	# Override so that we can control split.
	def init_dataset(self, split):
		torch.manual_seed(0)
		self.dataloader = MIME.data_loader(self.opts, split=split)

	def evaluate(self):
			
		z_sample = torch.zeros(1, self.opts.nz).uniform_() - 0.5
		z_sample = z_sample.cuda()

		trj, probs, samples = self.model.primitive_decoder(z_sample)
		embed()

def main(_):
	opts = flags.FLAGS
	
	if opts.st_space == 'joint_both':
		opts.n_state = 14
		print("Setting state to 14.")
	if opts.st_space =='joint_both_gripper':
		opts.n_state = 16
		print("Setting state to 16.")

	opts.logging_dir = osp.join(opts.logging_dir, 'mime')
	torch.manual_seed(0)

	# Create evaluator instance. 
	evaluator = PrimitiveDiscoverEvaluator(opts)

	# Set up testing. 
	evaluator.setup_testing(split='val')
	
	evaluator.evaluate()



	# PASS THE SKILL NET TO GYM ENVIRONMENT using this object
	# evaluator.skill_network



if __name__ == '__main__':
	app.run(main)