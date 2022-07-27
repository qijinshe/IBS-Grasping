"""
   Divide IBS into many parts
   Decompose Q Function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import tensorwatch as tw


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-10

joints = 24
point_num = 4096
channel = 18


def weights_init_(m): # Initialize Policy weights
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    return m


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.local_net = nn.Sequential(
            torch.nn.Conv1d(channel-5, 64, 1), nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1), nn.ReLU(),
            torch.nn.Conv1d(128, 128, 1), 
        )
        self.global_net = nn.Sequential(
            torch.nn.Conv1d(channel-5, 64, 1), nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1), nn.ReLU(),
            torch.nn.Conv1d(128, 512, 1), 
        )
    
    def mask_maxpooling(self, lx, mask):
        default_value = 0
        mask0 = (torch.sum(mask, dim=1, keepdim=True)<0.9)
        mask1 = mask[:,0,:].bool().unsqueeze(1).expand([-1,128,-1])
        mask2 = mask[:,1,:].bool().unsqueeze(1).expand([-1,128,-1])
        mask3 = mask[:,2,:].bool().unsqueeze(1).expand([-1,128,-1])
        mask4 = mask[:,3,:].bool().unsqueeze(1).expand([-1,128,-1])
        mask5 = mask[:,4,:].bool().unsqueeze(1).expand([-1,128,-1])
        ##########
        lx0 = torch.max(lx.masked_fill(~mask0, default_value), 2, keepdim=True)[0]
        lx1 = torch.max(lx.masked_fill(~mask1, default_value), 2, keepdim=True)[0]
        lx2 = torch.max(lx.masked_fill(~mask2, default_value), 2, keepdim=True)[0]
        lx3 = torch.max(lx.masked_fill(~mask3, default_value), 2, keepdim=True)[0]
        lx4 = torch.max(lx.masked_fill(~mask4, default_value), 2, keepdim=True)[0]
        lx5 = torch.max(lx.masked_fill(~mask5, default_value), 2, keepdim=True)[0]
        ##########
        rx0 = torch.min(lx.masked_fill(~mask0, default_value), 2, keepdim=True)[0]
        rx1 = torch.min(lx.masked_fill(~mask1, default_value), 2, keepdim=True)[0]
        rx2 = torch.min(lx.masked_fill(~mask2, default_value), 2, keepdim=True)[0]
        rx3 = torch.min(lx.masked_fill(~mask3, default_value), 2, keepdim=True)[0]
        rx4 = torch.min(lx.masked_fill(~mask4, default_value), 2, keepdim=True)[0]
        rx5 = torch.min(lx.masked_fill(~mask5, default_value), 2, keepdim=True)[0]
        ##########
        lx0 = lx0.view(-1,128)
        lx1 = lx1.view(-1,128)
        lx2 = lx2.view(-1,128)
        lx3 = lx3.view(-1,128)
        lx4 = lx4.view(-1,128)
        lx5 = lx5.view(-1,128)
        ##########
        rx0 = rx0.view(-1,128)
        rx1 = rx1.view(-1,128)
        rx2 = rx2.view(-1,128)
        rx3 = rx3.view(-1,128)
        rx4 = rx4.view(-1,128)
        rx5 = rx5.view(-1,128)
        return [lx0, lx1, lx2, lx3, lx4, lx5, rx0, rx1, rx2, rx3, rx4, rx5]

    def forward(self, points):
        num = (points.shape[1]//channel)
        points = points.reshape((-1,channel,num))
        x, mask = torch.split(points,[channel-5,5],dim=1)
        ##########
        gx = self.global_net(x) # Global_Features
        gx = torch.max(gx, 2, keepdim=True)[0]
        gx = gx.view(-1, 512)
        ##########
        lx = self.local_net(x) # Local_Features
        lx = self.mask_maxpooling(lx, mask)
        ##########
        x = torch.cat([*lx,gx],dim=-1) # Global_Local
        return x


class QMLP(nn.Module):
    def __init__(self, hidden_dim):
        super(QMLP, self).__init__()
        self.q_p1 = nn.Sequential(
            weights_init_(nn.Linear(2048+256+256, 2*hidden_dim)), nn.ReLU(), 
            weights_init_(nn.Linear(2*hidden_dim, hidden_dim)), nn.ReLU(),
            weights_init_(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            weights_init_(nn.Linear(hidden_dim, 2)),
        )
        self.q_p2 = nn.Sequential(
            weights_init_(nn.Linear(2048+256+256, 2*hidden_dim)), nn.ReLU(), 
            weights_init_(nn.Linear(2*hidden_dim, hidden_dim)), nn.ReLU(),
            weights_init_(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            weights_init_(nn.Linear(hidden_dim, 6)),
        ) 
    
    def forward(self, vector):
        x_p1 = self.q_p1(vector)
        x_p2 = self.q_p2(vector)
        x = torch.cat([x_p1[:,:1], x_p2, x_p1[:,1:2]], 1)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.pnet = Encoder()
        self.hidden1 = QMLP(hidden_dim)
        self.hidden2 = QMLP(hidden_dim)
        self.joint_state = joints
        self.state_extend = nn.Sequential(
            weights_init_(nn.Linear(self.joint_state, 256)),
        )
        self.action_extend = nn.Sequential(
            weights_init_(nn.Linear(num_actions+1, 256)), # Action+Stop
        )

    def forward(self, state, action, stop):
        stop = stop * 0.025
        x_a = state[:,:-self.joint_state]
        x_b = state[:,-self.joint_state:]
        x_a = self.pnet(x_a)
        x_b = self.state_extend(x_b)
        x_c = torch.cat([action, stop], 1)
        x_c = self.action_extend(x_c)
        xu = torch.cat([x_a, x_b, x_c], 1)
        x1 = self.hidden1(xu)
        x2 = self.hidden2(xu)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.pnet = Encoder()
        self.joint_state = joints
        self.hidden = nn.Sequential(
            weights_init_(nn.Linear(2048+256, 2*hidden_dim)), nn.ReLU(), 
            weights_init_(nn.Linear(2*hidden_dim, hidden_dim)), nn.ReLU(),
            weights_init_(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
        )
        self.hidden_stop = nn.Sequential(
            weights_init_(nn.Linear(2048+256, 2*hidden_dim)), nn.ReLU(), 
            weights_init_(nn.Linear(2*hidden_dim, hidden_dim)), nn.ReLU(),
            # weights_init_(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
        )
        self.state_extend = nn.Sequential(
            weights_init_(nn.Linear(self.joint_state, 256)),
        )
        ##########
        self.mean_linear = weights_init_(nn.Linear(hidden_dim, num_actions)) # predict mean value
        self.log_std_linear = weights_init_(nn.Linear(hidden_dim, num_actions)) # predict variance
        self.mean_stop = weights_init_(nn.Linear(hidden_dim, 1))
        self.log_std_stop = weights_init_(nn.Linear(hidden_dim, 1))
        ##########
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else: # action rescaling
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low)/2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low)/2.)

    def forward(self, state):
        x_a = state[:,:-self.joint_state]
        x_b = state[:,-self.joint_state:]
        x_a = self.pnet(x_a)
        x_b = self.state_extend(x_b)
        x = torch.cat([x_a, x_b], 1)
        x_act = self.hidden(x)
        x_stop = self.hidden_stop(x)
        ##########
        mean = self.mean_linear(x_act)
        mean_stop = self.mean_stop(x_stop)
        log_std = self.log_std_linear(x_act)
        log_std_stop = self.log_std_stop(x_stop)
        ##########
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        log_std_stop = torch.clamp(log_std_stop, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, mean_stop, log_std_stop

    def sample(self, state):
        mean, log_std, mean_stop, log_std_stop = self.forward(state)
        ##########
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t) # Enforcing Action Bound
        log_prob = normal.log_prob(x_t)
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob -= torch.log((1-y_t.pow(2))+epsilon) 
        log_prob = log_prob.sum(1, keepdim=True)
        ##########
        std_stop = log_std_stop.exp()
        normal_stop = Normal(mean_stop, std_stop)
        x_stop = normal_stop.rsample()
        y_stop = torch.tanh(x_stop)
        log_stop = normal_stop.log_prob(x_stop)
        log_stop -= torch.log((1-y_stop.pow(2))+epsilon)
        log_stop = log_stop.sum(1, keepdim=True)
        ##########
        stop = y_stop
        mean_stop = torch.tanh(mean_stop)
        action = (y_t * self.action_scale) + self.action_bias
        mean = (torch.tanh(mean) * self.action_scale) + self.action_bias
        return action, log_prob, mean, stop, log_stop, mean_stop

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)