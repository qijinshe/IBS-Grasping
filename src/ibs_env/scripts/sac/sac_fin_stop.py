import os
import time
import torch
import numpy as np
from functools import reduce
import torch.nn.functional as F
from torch.optim import Adam, SGD

from .utils import soft_update, hard_update
from .model_stop import GaussianPolicy, QNetwork


import sys
sys.path.append('..')
use_selfColl = True
try:
    from selfCollision import SelfCollision
except:
    use_selfColl = False


class SAC_FIN_STOP(object):
    policy_module = GaussianPolicy
    critic_module = QNetwork

    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda:0")
        self.self_counter = 0
        self.use_selfColl = use_selfColl
        self.action_dim = action_space.shape[0] 
        ###
        self.critic = self.critic_module(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = self.critic_module(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        ### Self-Collision Term (from [])
        if self.use_selfColl:
            SelfCollision.buffet(action_space.shape[0])
        ###
        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning is True:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper # auto entropy
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = SGD([self.log_alpha], lr=args.lr/10, momentum=0)
        self.policy = self.policy_module(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, log_pi, _, spi, log_spi, _ = self.policy.sample(state) # 0625
        else:
            _, log_pi, action, _, log_spi, spi = self.policy.sample(state)
        return action.detach().cpu().numpy()[0], spi.detach().cpu().numpy()[0]
    
        
    def evaluateQ(self, state, action, stop):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        stop = torch.FloatTensor(stop).to(self.device).unsqueeze(0)
        value, _ = self.critic(state, action, stop)
        return value.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, all_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(all_batch[:,:self.action_dim]).to(self.device) # Joint changes
        stop_batch = torch.FloatTensor(all_batch[:,self.action_dim:]).to(self.device) # Terminate value
        # Move to Graphic Card (if avaiable)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device) 
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device) # A vectorized mask

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, next_state_stop, next_state_log_spi, _ = self.policy.sample(next_state_batch) # maxQ'
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, next_state_stop) # maxQ'
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            ###
            next_q_value = torch.zeros((min_qf_next_target.shape[0],8)).to("cuda")
            next_q_value[:,:-1] = reward_batch + mask_batch * self.gamma * min_qf_next_target[:,:-1] # Grasp Reward ([0]) and Reaching Reward ([1:7])
            next_q_value[:,-1:] = mask_batch[:,0:1] * self.gamma * (min_qf_next_target[:,-1:] - self.alpha * (next_state_log_pi+next_state_log_spi)) # Entropy

        ### Critic Optimization
        qf1, qf2 = self.critic(state_batch, action_batch, stop_batch) # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = reduce(lambda x, y: x+y, [F.mse_loss(qf1[:,i], next_q_value[:,i]) for i in range(8)])
        qf2_loss = reduce(lambda x, y: x+y, [F.mse_loss(qf2[:,i], next_q_value[:,i]) for i in range(8)])
        qf_loss = (qf1_loss + qf2_loss)
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        ### Policy Optimization
        pi, log_pi, _ , spi, log_spi, _ = self.policy.sample(state_batch)
        # pi, log_pi, _ , _, log_spi, spi = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi, spi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (self.alpha * (log_pi+log_spi) - torch.sum(min_qf_pi, dim=-1)).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))] # It means the (Q_currnet - current * entropy) -> KL
        coll = torch.tensor(0)
        self.policy_optim.zero_grad()
        # Using SelfCollision Loss to reduce self-collision
        if self.self_counter % 10 == 0 and self.use_selfColl:
            policy_loss.backward(retain_graph=True)
            coll = SelfCollision.calculate(state_batch[:,-self.action_dim:]+pi).mean() * 100
            coll.backward()
        else:
            policy_loss.backward()
        self.policy_optim.step()
        self.self_counter += 1
        ### Entropy parameter Optimization
        if self.automatic_entropy_tuning:
            # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha_loss = -(self.log_alpha * (log_pi + log_spi + self.target_entropy).detach()).mean() # 1101
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), coll.item(), None, None, None, None, None

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        pre_path = "models/sac_pre_{}_{}".format(env_name, suffix)
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
