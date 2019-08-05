from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from absl import app
from absl import flags

import os.path as osp
import numpy as np
import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import Transformer
from ...nnutils import net_blocks as nb
from IPython import embed
import gc
from memory_profiler import profile

flags.DEFINE_integer('nz', 32, 'Dimension of high level skill space')
flags.DEFINE_integer('nh', 32, 'Dimension of hidden space in LSTMs')

flags.DEFINE_boolean('variable_nseg', True, 'Whether the latent prediction model unrolls a variable number of steps')
flags.DEFINE_boolean('variable_ns', True, 'Whether the skill net unrolls a variable number of steps')
flags.DEFINE_integer('fixed_prim_len', 10, 'If variable ns is false, use this prim len')
flags.DEFINE_integer('n_skill_segments', 4, 'Number of skill segments (used if variable_nseg=False)')
flags.DEFINE_integer('prim_len', 10, 'If > 0, then all primitives are of this length')
flags.DEFINE_float('dec_p_bias', 0., 'Bias for continuation probability in primitive decoder')
flags.DEFINE_float('lpred_p_bias', 0., 'Bias for continuation probability in latent predictor')


class TrajectoryEncoder(torch.nn.Module):
    def __init__(self, opts, reverse=True):
        super(TrajectoryEncoder, self).__init__()
        self.reverse = reverse
        self.lstm = torch.nn.LSTM(opts.n_state, opts.nh, 4)

    def forward(self, x):
        if self.reverse:
            x = torch.flip(x, [0])
        return self.lstm.forward(x)

class TrajectoryEncoderVAE(torch.nn.Module):
    def __init__(self, opts, reverse=True):
        super(TrajectoryEncoderVAE, self).__init__()
        self.reverse = reverse
        self.lstm = torch.nn.LSTM(opts.n_state, opts.nh*2, 4)

    def forward(self, x):
        if self.reverse:
            x = torch.flip(x, [0])
        return self.lstm.forward(x)

class PrimitiveDecoder(torch.nn.Module):
    def __init__(self, opts, max_prim_len=100):
        '''
        Args:
            opts: cnn specifications
            max_prim_len: stop primitive after this length
        '''
        super(PrimitiveDecoder, self).__init__()
        self.lstm = torch.nn.LSTM(opts.nz, opts.nh, 4)
        self.prob_predictor = torch.nn.Linear(opts.nh, 2)
        self.state_predictor = torch.nn.Linear(opts.nh, opts.n_state)
        self.max_prim_len = max_prim_len
        self.p_bias = opts.dec_p_bias

    def forward(self, z, max_unroll=False):
        '''
        Args:
            z: 1 X nz vector
        Returns states etc. for one sampled trajectory:
            states: T X 1 X n_state vectors
            probs: T X 1 X 2 termination probs
            samples: T X 1 samples (last one should be 1)
            if max_unroll = True:
                the states and probs are sampled until self.max_prim_len
                the assumption is that the outer loss accounts for the probabilistic ditribution
        '''
        states = []
        probs = []
        samples = []
        t = 0

        stop = 0
        inp = z.view(1,1,-1)
        zeros_inp = torch.zeros_like(inp)

        while stop != 1:
            if t == 0:
                o_t, h_t = self.lstm(inp)
            else:
                o_t, h_t = self.lstm(zeros_inp, h_t)
            t = t+1

            o_t = o_t[0]

            s_t = self.state_predictor(o_t)
            p_t = self.prob_predictor(o_t)

            p_t[:, 0] += self.p_bias #bias for continuing
            p_t = torch.nn.functional.softmax(p_t, dim=1)
            if t > self.max_prim_len:
                p_t = p_t*0
                p_t[:, 1] += 1

            sample_t = torch.distributions.Categorical(probs=p_t).sample()
            if max_unroll and t <= self.max_prim_len:
                sample_t = sample_t*0

            stop = (sample_t == 1)

            states.append(s_t)
            probs.append(p_t)
            samples.append(sample_t)

        states = torch.stack(states)
        probs = torch.stack(probs)
        samples = torch.stack(samples)
        
        return states, probs, samples

class PrimitiveDecoderKnownLength(torch.nn.Module):
    def __init__(self, opts, prim_len=10):
        '''
        Args:
            opts: cnn specifications
            max_prim_len: stop primitive after this length
        '''
        super(PrimitiveDecoderKnownLength, self).__init__()
        self.lstm = torch.nn.LSTM(opts.nz, opts.nh, 4)
        self.state_predictor = torch.nn.Linear(opts.nh, opts.n_state)
        self.prim_len = prim_len

    def forward(self, z, prim_len=None):
        '''
        Args:
            z: 1 X nz vector
        Returns states etc. the trajectory of length prim_len:
            (states): T X 1 X n_state vectors
        '''
        states = []
        probs = []
        samples = []
        t = 0

        if prim_len is None:
            prim_len = self.prim_len

        stop = 0
        inp = z.view(1,1,-1)
        zeros_inp = torch.zeros_like(inp)

        while t < prim_len:
            if t == 0:
                o_t, h_t = self.lstm(inp)
            else:
                o_t, h_t = self.lstm(zeros_inp, h_t)
            t = t+1

            o_t = o_t[0]
            s_t = self.state_predictor(o_t)

            states.append(s_t)

        states = torch.stack(states)

        return states, None, None

class ScoreFunctionEstimator(object):
    def __init__(self, use_baseline=True, momentum=0.9):
        self.b = 0
        self.use_b = use_baseline
        self.m = momentum

    def forward(self, score, probs, samples, update_b=True):
        '''
        Args:
            score: scalar return for the episode, or T X 1 return vector
            probs: T X 1 X nChoice distribution
            samples: T X 1 index vector
            update_b: Whether this call should be used to update basline
        Returns:
            l: A pseudo-loss whose derivative is equivalent to SF estimator
        '''
        score = score.detach()
        if self.use_b and update_b:
            self.b = self.m*self.b + (1-self.m)*score.mean().item()

        log_probs = torch.log(torch.gather(probs, dim=2, index=samples.view(-1,1,1)))
        log_probs = log_probs.view(-1, 1)
        score = score - self.b

        # pseudo_loss = torch.mean(log_probs)*score
        pseudo_loss = torch.sum(log_probs*score)
        return pseudo_loss

class Imgs2TrajectoryPredictor(torch.nn.Module):
    def __init__(self, opts):
        super(Imgs2TrajectoryPredictor, self).__init__()

        self.resnet = nb.ResNetConv()
        nz_img = 512*2*4*3
        self.encoder_img = nb.fc_stack(nz_img, nz_img//16, 2)
        self.encoder_trj = nb.fc_stack(nz_img//16, opts.nz, 2)
        self.seq_predictor = PrimitiveDecoder(opts)


    def forward(self, imgs):
        imgs_inp = imgs.permute(0,3,1,2).float()/255
        imgs_feat = self.resnet.forward(imgs_inp)
        imgs_feat = imgs_feat.view(1, -1)

        imgs_enc = self.encoder_img.forward(imgs_feat)
        z_seq = self.encoder_trj.forward(imgs_enc)

        return self.seq_predictor(z_seq)

# There is a class definition in .Transformer called Transformer()
# We are going to use that to create another IntendedTrajectoryPredictorTransformer (using a transformer). 

class IntendedTrajectoryPredictorTransformer(torch.nn.Module):

    def __init__(self, opts):
        super(IntendedTrajectoryPredictorTransformer, self).__init__()

        self.opts = opts 

        # Instantiate a transformer.
        if self.opts.variable_nseg:
            self.transformer = Transformer.TransformerVariableNSeg(opts, dummy_inputs=False).cuda()
        else:
            self.transformer = Transformer.TransformerFixedNSeg(opts, dummy_inputs=False).cuda()

        if self.opts.variable_ns:
            self.primitive_decoder = PrimitiveDecoder(opts)
        else:
            self.primitive_decoder = PrimitiveDecoderKnownLength(opts, prim_len=opts.fixed_prim_len)

    def forward(self, observed_trajectory, fake_fnseg=False):
        # (1) Use the transformer to predict Z's from the observed_trajectory. 
        # (2) Feed these Z's to the primitive decoder. 
        # (3) Return intended trajectory and other probabilities and stuff. 
        observed_trajectory = observed_trajectory.squeeze(1)

        if self.opts.variable_nseg:
            # Feed into transformer. 
            latent_z_seq, stop_probabilities = self.transformer.forward(observed_trajectory, fake_fnseg=fake_fnseg)
            # Stops were obviously 0's until the last element.
            sampled_stops = torch.zeros((latent_z_seq.shape[0])).cuda().long()
            sampled_stops[-1] += 1

            stop_probabilities = stop_probabilities.unsqueeze(1)
            sampled_stops = sampled_stops.unsqueeze(1)
        else:
            latent_z_seq = self.transformer.forward(observed_trajectory)
            stop_probabilities, sampled_stops = None, None

        prmitives = [self.primitive_decoder(z) for z in latent_z_seq]
        self.primitives = prmitives
        self.latent_z_seq = latent_z_seq

        traj_intended = torch.cat([prim[0] for prim in prmitives])

        if self.opts.variable_ns:
            probs = torch.cat([prim[1] for prim in prmitives])
            samples = torch.cat([prim[2] for prim in prmitives])
        else:
            probs, samples = None, None        
        return traj_intended, probs, samples, stop_probabilities, sampled_stops

class IntendedTrajectoryPredictorTransformerVAE(torch.nn.Module):

    def __init__(self, opts):
        super(IntendedTrajectoryPredictorTransformerVAE, self).__init__()

        self.opts = opts 

        # Instantiate a transformer.
        if self.opts.variable_nseg:
            self.transformer = Transformer.TransformerVariableNSeg(opts, dummy_inputs=False).cuda()
        else:
            self.transformer = Transformer.TransformerFixedNSeg(opts, dummy_inputs=False).cuda()

        if self.opts.variable_ns:
            self.primitive_decoder = PrimitiveDecoder(opts)
        else:
            self.primitive_decoder = PrimitiveDecoderKnownLength(opts, prim_len=opts.fixed_prim_len)

        # Create linear layer to split prediction into mu and sigma. 
        self.mu_linear_layer = torch.nn.Linear(self.opts.nz, self.opts.nz)
        self.sig_linear_layer = torch.nn.Linear(self.opts.nz, self.opts.nz)

    # @profile
    def forward(self, observed_trajectory, fake_fnseg=False):
        # (1) Use the transformer to predict Z's from the observed_trajectory. 
        # (2) Feed these Z's to the primitive decoder. 
        # (3) Return intended trajectory and other probabilities and stuff. 
        observed_trajectory = observed_trajectory.squeeze(1)

        if self.opts.variable_nseg:
            # Feed into transformer. 
            transformer_output, stop_probabilities = self.transformer.forward(observed_trajectory, fake_fnseg=fake_fnseg)
            latent_z_mu_seq = self.mu_linear_layer(transformer_output)
            latent_z_log_sig_seq = self.sig_linear_layer(transformer_output)

            # Stops were obviously 0's until the last element.
            sampled_stops = torch.zeros((latent_z_mu_seq.shape[0])).cuda().long()
            sampled_stops[-1] += 1

            stop_probabilities = stop_probabilities.unsqueeze(1)
            sampled_stops = sampled_stops.unsqueeze(1)
        else:
            # latent_z_seq_param = self.linear_layer(self.transformer.forward(observed_trajectory, fake_fnseg=fake_fnseg))
            transformer_output = self.transformer.forward(observed_trajectory, fake_fnseg=fake_fnseg)
            latent_z_mu_seq = self.mu_linear_layer(transformer_output)
            latent_z_log_sig_seq = self.sig_linear_layer(transformer_output)

            stop_probabilities, sampled_stops = None, None

        # Compute standard deviation as exponentiated log sigma. 
        std = torch.exp(0.5*latent_z_log_sig_seq)
        # Sample random variable from uniform. 
        eps = torch.randn_like(std)
        
        # Compute Zs as mu+eps*std
        self.latent_z_seq = latent_z_mu_seq+eps*std
        
        # # Instead of using a list for primitives, we know that variable_ns is False. So just maintain a tensor for trajectory intended. 
        # self.primitives = None

        # Compute KL Divergence Loss term here, so we don't have to return mu's and sigma's. 
        self.kld_loss = 0.
        for t in range(latent_z_mu_seq.shape[0]):
            # Taken from mime_plan_skill.py Line 159 - KL Divergence for Gaussian prior and Gaussian prediction. 
            self.kld_loss += float(-0.5 * torch.sum(1. + latent_z_log_sig_seq[t] - latent_z_mu_seq[t].pow(2) - latent_z_log_sig_seq[t].exp()))
    
        self.primitives = [self.primitive_decoder(z) for z in self.latent_z_seq]
        traj_intended = torch.cat([prim[0] for prim in self.primitives])

        if self.opts.variable_ns:
            probs = torch.cat([prim[1] for prim in self.primitives])
            samples = torch.cat([prim[2] for prim in self.primitives])
        else:
            probs, samples = None, None

        # gc.collect()

        return traj_intended, probs, samples, stop_probabilities, sampled_stops