#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from absl import flags

sys.path.append('/private/home/shubhtuls/code')


from sfd.nnutils import train_utils
from sfd.utils import baxter_vis
from sfd.Experiments.abstraction import abstraction_utils

flags.FLAGS([''])
# flags.DEFINE_integer('n_state', 16, 'Dimension of state space. Set automatically based on state space')
opts = flags.FLAGS

baxter_visualizer = baxter_vis.MujocoVisualizer()

# pretrain_skillnet_name = 'mime_plan_ra_fns'; opts.nz = 64; opts.nh = 64;
# num_pretrain_skillnet_epoch = 80
# skill_net = abstraction_utils.PrimitiveDecoderKnownLength(opts)

opts.transformer = True
opts.variable_nseg = False
opts.variable_ns = False
opts.nz = 64
opts.nh = 64
opts.st_space=joint_both_gripper
opts.n_state = 16 

full_model = abstraction_utils.IntendedTrajectoryPredictorTransformer(opts)
trainer = train_utils.Trainer(opts)

network_dir = "Path to T313 model...."

trainer.load_network(full_model, 'pred', 'latest', network_dir=network_dir)


z_sample = torch.zeros(1, opts.nz).uniform_() - 0.5

trj, probs, samples = full_model.primitive_decoder.forward(z_sample)

trj = trj.detach().cpu().numpy()
for t in range(trj.shape[0]):
    img = baxter_visualizer.set_joint_pose_return_image(trj[t,0], arm='right', gripper=False)
    plt.imshow(img)
    plt.show()
