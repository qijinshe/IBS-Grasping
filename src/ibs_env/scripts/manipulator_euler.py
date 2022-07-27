"""
    0218 Manipulator
"""
from hand import Hand
import numpy as np
import gym
from scipy.spatial.transform import Rotation as R


class HandManipulator(object):

    @property
    def hand_center(self):
        r = R.from_euler('xyz', self.state[3:6])
        # m = r.as_dcm() # scipy<1.4
        m = r.as_matrix() # scipy>=1.4
        offset = np.array([1,0,0])
        return self.state[:3]+m@offset

    def set_action_space(self, action_dim):
        SCALE = np.ones(action_dim) * 0.025
        self.action_LB = -SCALE
        self.action_UB = SCALE
        self.SCALE = SCALE
        self.action_space = gym.spaces.Box(low=self.action_LB, high=self.action_UB, dtype=np.float32)

    def __init__(self, path, action_dim):
        self.action_dim = action_dim
        self.hand = Hand(path, 0.01, use_joint_limit=False, use_quat=True, use_eigen=False) # SCALE
        self.hand.sample(13000)
        # self.hand.sample(13000, re_sample=True)
        lb, ub = self.hand.lb_ub()
        self.joints_LB = np.array(lb)
        self.joints_UB = np.array(ub)
        self.set_action_space(action_dim)
        self.state = None
        self.rot_pose = None

    def bound_state(self, action):
        full_action = action.copy()
        new_state = (self.state+full_action)
        new_state[6:] = np.clip(new_state[6:], self.joints_LB, self.joints_UB) # The bound of internal joints
        # The bound of Euler Agnle (RYP) ([-pi,pi],[-pi/2,pi/2],[-pi,pi])
        new_state[3] = (new_state[3]+np.pi)%(2*np.pi) - np.pi
        new_state[4] = np.clip(new_state[4],-np.pi/2, np.pi/2)
        new_state[5] = (new_state[5]+np.pi)%(2*np.pi) - np.pi
        return new_state
    
    def execute(self, action):
        self.state = self.bound_state(action)
        new_quat = self._euler2quat(self.state[3:6])
        ex_dofs = np.array([*self.state[:3], *new_quat])
        self.hand.forward_kinematics(ex_dofs, self.state[6:])
        return self.get_euler_state()
        
    def reset(self, joints):
        if joints.shape[0] == self.action_dim:
            self.state = joints.copy()
            ex_dofs = np.zeros(7)
            ex_dofs[:3] = self.state[:3]
            ex_dofs[3:7] = self._euler2quat(self.state[3:6])
            ######
            self.hand.forward_kinematics(ex_dofs, self.state[6:])
        else:
            self.state = np.zeros(self.action_dim)
            self.state[:3] = joints[:3]
            self.state[3:6] = R.from_quat([*joints[4:7], joints[3]]).as_euler('xyz')
            self.state[6:] = joints[7:]
            ######
            self.hand.forward_kinematics(joints[:7], self.state[6:])
        return self.getPointCloud()

    def get_euler_state(self):
        # Dim 24 (3Translation + 3Rotation + 18DoA)
        ret = self.state.copy()
        return ret
    
    def get_quat_state(self):
        # Dim:25 (3Translation + 4Rotation + 18DoA)
        ret = np.zeros(25)
        ret[7:] = self.state[6:]
        ret[:3] = self.state[:3]
        ret[3:7] = self._euler2quat(self.state[3:6])
        return ret

    def get_state(self): # old entry
        ret =  self.state.copy()
        ret[:3] = 0
        ret[3:6] = (ret[3:6]/np.pi) # remove pi1
        return ret
    
    def get_modified_state(self): # stop2
        ret =  self.state.copy()
        ret[:3] = 0
        ret[3:6] = (ret[3:6]/np.pi) # remove pi1
        ret[6:] = (ret[6:]/np.pi) # remove pi2
        return ret

    def getPointCloud(self):
        points = self.hand.sample_fwk()
        return points

    def getPointNormal(self):
        normals = self.hand.sample_fwk_normal()
        return normals

    def get_label(self, idx=None):
        if idx is not None:
            return self.hand.labels[idx.astype('int32')]
        else:
            return self.hand.labels.copy()

    def get_part_label(self, idx=None):
        if idx is not None:
            return self.hand.part_labels[idx.astype('int32')]
        else:
            return self.hand.part_labels.copy()
    
    def get_internal_label(self, idx=None):
        if idx is not None:
            return self.hand.internal_labels[idx.astype('int32')]
        else:
            return self.hand.internal_labels.copy()

    def get_offset(self, euler):
        if euler.shape[0] == 3:
            r = R.from_euler('xyz', euler)
        else:
            r = R.from_quat([euler[1],euler[2],euler[3],euler[0]])
        m = r.as_matrix() 
        offset = np.array([1,0,0])
        return m@offset
    
    def _euler2quat(self, euler):
        new_quat = R.from_euler('xyz', euler).as_quat()
        return [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

    def random_action(self):
        action = (2*np.random.rand(self.action_dim)-1) * self.SCALE
        return action

    def hacker_action(self, tar_dof, add_noise=False):
        if tar_dof is None:
            return self.random_action()
        action = np.zeros(self.action_dim)
        action[:3] = (tar_dof[:3]-self.state[:3])
        first = np.zeros(25)
        if np.sum(np.abs(action[:3])) < 0.02:
            action[-18:] = (tar_dof[-18:]-self.state[-18:])
        else:
            action[-18:] = (first[-18:]-self.state[-18:])
        if add_noise:
            noise = 0.1 * (2*np.random.rand(self.action_dim)-1) * self.SCALE
            action += noise
        action = np.clip(action, self.action_LB, self.action_UB)
        return action

