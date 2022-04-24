import math
import numpy as np
import torch
import torch.nn.functional as F
from a3c_utils import normal, process_state

import random

class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.done_1 = False
        self.actual_model = 1
        self.model2 = None
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.states = []

        random.seed(10)
        torch.manual_seed(10)

    def action_train(self):
        if self.args.model == 'CONV' or self.args.model == 'CONV_2':
            if self.args.frame_history>1 or self.args.TwoDim or (self.args.velocities and not self.args.vel_interp) or self.args.PC_vel:
                self.state = self.state.squeeze().unsqueeze(0)
            else:
                self.state = self.state.unsqueeze(0).unsqueeze(0)

            if self.args.lstm:
                value, mu, sigma, (self.hx, self.cx) = self.model(
                    (self.state, (self.hx, self.cx)))
            else:
                value, mu, sigma = self.model(self.state)
        mu = torch.clamp(mu, -1.0, 1.0) # cambiar a softsign o tanh
        sigma = F.softplus(sigma) + 1e-5
        eps = torch.randn(mu.size())
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                eps = eps.cuda()
                pi = pi.cuda()
        else:
            eps = eps
            pi = pi

        action = (mu + sigma.sqrt() * eps).data
        act = action
        prob = normal(act, mu, sigma, self.gpu_id, gpu=self.gpu_id >= 0)
        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        self.entropies.append(entropy)
        log_prob = (prob + 1e-6).log()
        self.log_probs.append(log_prob)
        state, reward, self.done, self.info = self.env.step(
            action.cpu().numpy()[0])
        reward = max(min(float(reward), 1.0), -1.0)
        if self.args.frame_history > 1:
            self.states.append(state)
            self.state = process_state(torch.as_tensor(self.states).float(),velocities=self.args.velocities,vel_interp=self.args.vel_interp)
        else:
            self.state = process_state(torch.from_numpy(state).float(),velocities=self.args.velocities,vel_interp=self.args.vel_interp)

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        # if self.model2 is not None:
        #     if self.actual_model == 1:
        #         self.done_1 = self.eps_len >= (self.args.max_episode_length - self.args.max_episode_length2)
        #         if self.done_1:
        #             self.actual_model = 2
        #             self.env.action_dist_step = self.args.dist_step2
        #             self.env.action_angle_step = self.args.angle_step2
        #             self.env.scale_d = self.args.scale_d2
        # else:
        #     self.done_1 = self.eps_len >= (self.args.max_episode_length - self.args.max_episode_length2)
        #     if self.done_1:
        #         self.env.action_dist_step = self.args.dist_step2
        #         self.env.action_angle_step = self.args.angle_step2
        #         self.env.scale_d = self.args.scale_d2
        #     else:
        #         self.env.action_dist_step = self.args.dist_step
        #         self.env.action_angle_step = self.args.angle_step
        #         self.env.scale_d = self.args.scale_d
        self.done = self.done or self.eps_len >= self.args.max_episode_length

        self.values.append(value)
        self.rewards.append(reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.args.lstm:
                    if self.gpu_id >= 0:
                        with torch.cuda.device(self.gpu_id):
                            self.cx = torch.zeros(
                                1, self.args.out_feats).cuda()
                            self.hx = torch.zeros(
                                1, self.args.out_feats).cuda()
                    else:
                        self.cx = torch.zeros(1, self.args.out_feats)
                        self.hx = torch.zeros(1, self.args.out_feats)
            else:
                if self.args.lstm:
                    self.cx = self.cx.data
                    self.hx = self.hx.data
            if self.args.model == 'CONV' or self.args.model == 'CONV_2':
                if self.args.frame_history>1 or self.args.TwoDim or (self.args.velocities and not self.args.vel_interp) or self.args.PC_vel:
                    self.state = self.state.squeeze().unsqueeze(0)
                else:
                    self.state = self.state.unsqueeze(0).unsqueeze(0)
                if self.args.lstm:
                    value, mu, sigma, (self.hx, self.cx) = self.model(
                        (self.state, (self.hx, self.cx)))
                else:
                    value, mu, sigma = self.model(self.state)
        mu = torch.clamp(mu.data, -1.0, 1.0)
        action = mu.cpu().numpy()[0]
        state, self.reward, self.done, self.info = self.env.step(action)
        if self.args.frame_history > 1:
            self.states.append(state)
            self.state = process_state(torch.as_tensor(self.states).float(),velocities=self.args.velocities,vel_interp=self.args.vel_interp)
        else:
            self.state = process_state(torch.from_numpy(state).float(),velocities=self.args.velocities,vel_interp=self.args.vel_interp)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1

        # if self.model2 is not None:
        #     if self.actual_model == 1:
        #         self.done_1 = self.eps_len >= (self.args.max_episode_length - self.args.max_episode_length2)
        #         if self.done_1:
        #             self.actual_model = 2
        #             self.env.action_dist_step = self.args.dist_step2
        #             self.env.action_angle_step = self.args.angle_step2
        #             self.env.scale_d = self.args.scale_d2
        # else:
        #     self.done_1 = self.eps_len >= (self.args.max_episode_length - self.args.max_episode_length2)
        #     if self.done_1:
        #         self.env.action_dist_step = self.args.dist_step2
        #         self.env.action_angle_step = self.args.angle_step2
        #         self.env.scale_d = self.args.scale_d2
        #     else:
        #         self.env.action_dist_step = self.args.dist_step
        #         self.env.action_angle_step = self.args.angle_step
        #         self.env.scale_d = self.args.scale_d

        self.done = self.done or self.eps_len >= self.args.max_episode_length

        return self

    def action_test_full(self):
        with torch.no_grad():
            if self.done:
                if self.args.lstm:
                    if self.gpu_id >= 0:
                        with torch.cuda.device(self.gpu_id):
                            self.cx = torch.zeros(
                                1, self.args.out_feats).cuda()
                            self.hx = torch.zeros(
                                1, self.args.out_feats).cuda()
                    else:
                        self.cx = torch.zeros(1, self.args.out_feats)
                        self.hx = torch.zeros(1, self.args.out_feats)
            else:
                if self.args.lstm:
                    self.cx = self.cx.data
                    self.hx = self.hx.data
            if self.args.model == 'CONV' or self.args.model == 'CONV_2':
                if self.args.frame_history>1 or self.args.TwoDim or (self.args.velocities and not self.args.vel_interp) or self.args.PC_vel:
                    self.state = self.state.squeeze().unsqueeze(0)
                else:
                    self.state = self.state.unsqueeze(0).unsqueeze(0)
                if self.args.lstm:
                    #print(torch.max(self.state),torch.min(self.state))
                    value, mu, sigma, (self.hx, self.cx) = self.model(
                        (self.state, (self.hx, self.cx)))
                else:
                    value, mu, sigma = self.model(self.state)
        mu = torch.clamp(mu.data, -1.0, 1.0)
        action = mu.cpu().numpy()[0]
        state, self.reward, self.done, self.info = self.env.step(action)
        if self.args.frame_history > 1:
            self.states.append(state['VI'])
            self.state = process_state(torch.as_tensor(self.states).float(),velocities=self.args.velocities,vel_interp=self.args.vel_interp)
        else:
            self.state = process_state(torch.from_numpy(state['VI']).float(),velocities=self.args.velocities,vel_interp=self.args.vel_interp)
        #print(torch.max(self.state),torch.min(self.state))
        self.state_full = state
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        sigma = F.softplus(sigma) + 1e-5
        entropy = sigma
        #entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        self.entropies.append(entropy.squeeze().numpy())
        self.values.append(value.numpy())
        self.rewards.append(self.reward)


        # if self.model2 is not None:
        #     if self.actual_model == 1:
        #         self.done_1 = self.eps_len >= (self.args.max_episode_length - self.args.max_episode_length2)
        #         if self.done_1:
        #             self.actual_model = 2
        #             self.env.action_dist_step = self.args.dist_step2
        #             self.env.action_angle_step = self.args.angle_step2
        #             self.env.scale_d = self.args.scale_d2
        # else:
        #     self.done_1 = self.eps_len >= (self.args.max_episode_length - self.args.max_episode_length2)
        #     if self.done_1:
        #         self.env.action_dist_step = self.args.dist_step2
        #         self.env.action_angle_step = self.args.angle_step2
        #         self.env.scale_d = self.args.scale_d2
        self.done = self.done or self.eps_len >= self.args.max_episode_length

        self.action = action
        return self
    def action_test_full_sample(self):
        with torch.no_grad():
            if self.done:
                if self.args.lstm:
                    if self.gpu_id >= 0:
                        with torch.cuda.device(self.gpu_id):
                            self.cx = torch.zeros(
                                1, self.args.out_feats).cuda()
                            self.hx = torch.zeros(
                                1, self.args.out_feats).cuda()
                    else:
                        self.cx = torch.zeros(1, self.args.out_feats)
                        self.hx = torch.zeros(1, self.args.out_feats)
            else:
                if self.args.lstm:
                    self.cx = self.cx.data
                    self.hx = self.hx.data
            if self.args.model == 'CONV' or self.args.model == 'CONV_2':
                if self.args.frame_history>1 or self.args.TwoDim or (self.args.velocities and not self.args.vel_interp) or self.args.PC_vel:
                    self.state = self.state.squeeze().unsqueeze(0)
                else:
                    self.state = self.state.unsqueeze(0).unsqueeze(0)
                if self.args.lstm:
                    #print(torch.max(self.state),torch.min(self.state))
                    value, mu, sigma, (self.hx, self.cx) = self.model(
                        (self.state, (self.hx, self.cx)))
                else:
                    value, mu, sigma = self.model(self.state)

        mu = torch.clamp(mu, -1.0, 1.0) # cambiar a softsign o tanh
        sigma = F.softplus(sigma) + 1e-5
        eps = torch.randn(mu.size()) * 0.5
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                eps = eps.cuda()
                pi = pi.cuda()
        else:
            eps = eps
            pi = pi

        #action = (mu + sigma.sqrt() * eps).data
        if self.eps_len < 100:
            action = (mu + sigma.sqrt() * eps).data
        else:
            action = mu.data
        act = action
        prob = normal(act, mu, sigma, self.gpu_id, gpu=self.gpu_id >= 0)

        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        self.entropies.append(entropy.squeeze().numpy())
        log_prob = (prob + 1e-6).log()
        self.log_probs.append(log_prob)
        # print('prob',prob,prob[0].shape)
        # print('mu',mu,'sigma',sigma,'x',act)
        #state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy()[0])
        state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy()[0])
        if self.args.frame_history > 1:
            self.states.append(state['VI'])
            self.state = process_state(torch.as_tensor(self.states).float(),velocities=self.args.velocities,vel_interp=self.args.vel_interp)
        else:
            self.state = process_state(torch.from_numpy(state['VI']).float(),velocities=self.args.velocities,vel_interp=self.args.vel_interp)
        #print(torch.max(self.state),torch.min(self.state))
        self.state_full = state
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1

        self.values.append(value.numpy())
        self.rewards.append(self.reward)


        # if self.model2 is not None:
        #     if self.actual_model == 1:
        #         self.done_1 = self.eps_len >= (self.args.max_episode_length - self.args.max_episode_length2)
        #         if self.done_1:
        #             self.actual_model = 2
        #             self.env.action_dist_step = self.args.dist_step2
        #             self.env.action_angle_step = self.args.angle_step2
        #             self.env.scale_d = self.args.scale_d2
        # else:
        #     self.done_1 = self.eps_len >= (self.args.max_episode_length - self.args.max_episode_length2)
        #     if self.done_1:
        #         self.env.action_dist_step = self.args.dist_step2
        #         self.env.action_angle_step = self.args.angle_step2
        #         self.env.scale_d = self.args.scale_d2
        self.done = self.done or self.eps_len >= self.args.max_episode_length

        self.action = action
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        if self.done:
            if self.args.lstm:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = torch.zeros(
                            1, self.args.out_feats).cuda()
                        self.hx = torch.zeros(
                            1, self.args.out_feats).cuda()
                else:
                    self.cx = torch.zeros(1, self.args.out_feats)
                    self.hx = torch.zeros(1, self.args.out_feats)
        return self

    def clear_lstm(self):
        if self.args.lstm:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = torch.zeros(
                        1, self.args.out_feats).cuda()
                    self.hx = torch.zeros(
                        1, self.args.out_feats).cuda()
            else:
                self.cx = torch.zeros(1, self.args.out_feats)
                self.hx = torch.zeros(1, self.args.out_feats)
        return self
