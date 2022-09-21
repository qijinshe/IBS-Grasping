"""
    This file is modified from "" [https://github.com/columbia-ai-robotics/adagrasp/]
"""

import math
import os
import time
from types import new_class
from functools import reduce

from numpy.core.fromnumeric import choose

import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

import numpy as np
from numpy.core.defchararray import join

from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt

from .urdf_editor import UrdfEditor
from .utils import depth_to_pointcloud


'''
    TODO: find the reason "stepSimulation" is problematic in our simulation
'''


class Gripper(object):
    """
    A moving mount and a gripper.
    the mount has 6 joints:
        0: prismatic x;
        1: prismatic y;
        2: prismatic z;
        3: revolute x;
        4: revolute y;
        5: revolute z;
    the gripper is transfered from xml files
    origin file: hand.xml
    """
    def sleep_with_state_saving(self, oid, interval, label, saving=True):
        # step the simulation and record hand states and object states
        if not saving:
            time.sleep(interval)
            return
        sleep_count = (interval * 120)
        sleep_count = int(sleep_count)
        for c in range(sleep_count):
            time.sleep(1/120.0)
            obj_pos, obj_quat = self._bullet_client.getBasePositionAndOrientation(oid)
            s, v = self.get_hand_dof_values()
            self.ostates.append(obj_pos + obj_quat)
            self.hstates.append(s)
            self.labels.append(label)

    def get_state_data(self):
        # get buffered state data
        hstates = np.array(self.hstates)
        ostates = np.array(self.ostates)
        labels = np.array(self.labels)
        self.hstates.clear()
        self.ostates.clear()
        self.labels.clear()
        return hstates, ostates, labels

    def __init__(self, bullet_client, home_position):
        self._bullet_client = bullet_client
        self._home_position = np.array(home_position) # the base of the robot
        self._default_orientation = np.array([0, 0, 0])
        self.height = 0.5 # the Height of the object to be lifted
        self.hstates = []
        self.ostates = []
        self.labels = []

        
        # load mount
        self._gripper_parent_index = 6 # 3 Pos + 3 Rot
        urdfname = "gripper_model/combined.urdf"
        if not os.path.exists(urdfname):
            gripper_body_id = self._bullet_client.loadURDF("gripper_model/hand2.urdf", basePosition=self._home_position)
            mount_body_id = self._bullet_client.loadURDF('gripper_model/mount.urdf', basePosition=self._home_position, useFixedBase=True)
            ed_mount = UrdfEditor()
            ed_gripper = UrdfEditor()
            ed_mount.initializeFromBulletBody(mount_body_id, self._bullet_client._client)
            ed_gripper.initializeFromBulletBody(gripper_body_id, self._bullet_client._client)
            # combine mount and gripper new with a joint
            newjoint = ed_mount.joinUrdf(
                childEditor=ed_gripper,
                parentLinkIndex=self._gripper_parent_index,
                jointPivotXYZInParent=np.array([0, 0, 0]),
                jointPivotRPYInParent=np.array([0, 0, 0]),
                jointPivotXYZInChild=[0, 0, 0],
                jointPivotRPYInChild=[0, 0, 0],
                parentPhysicsClientId=self._bullet_client._client,
                childPhysicsClientId=self._bullet_client._client
            )
            newjoint.joint_type = self._bullet_client.JOINT_FIXED
            newjoint.joint_name = "joint_mount_gripper"

            ed_mount.saveUrdf(urdfname)
            # remove mount and gripper bodies
            self._bullet_client.removeBody(mount_body_id)
            self._bullet_client.removeBody(gripper_body_id)
        
        self._body_id = self._bullet_client.loadURDF(urdfname, useFixedBase=True, basePosition=self._home_position, 
            baseOrientation=self._bullet_client.getQuaternionFromEuler([0, 0, 0]))

        # The number of the controllable internal degrees of freedom
        self.hand_dofs = [
            8, 10, 12, 14, 
            18, 20, 22,
            26, 28, 30,
            34, 36, 38,
            42, 44, 46, 48, 50,
        ]
        self.ex_dofs = [0, 1, 2, 3, 4, 5] # the number of the external dofs 
        self.undofs = [16, 24, 32, 40] # the number of the follow-up joints

        print("ALL JOINT NUM: ", self._bullet_client.getNumJoints(self._body_id))

        # configure the gripper (e.g. friction)
        for i in range(self._gripper_parent_index+1, self._bullet_client.getNumJoints(self._body_id)):
            self._bullet_client.changeDynamics(self._body_id, i, lateralFriction=1.0, spinningFriction=1.0, rollingFriction=0.0001, frictionAnchor=True) # Friction
            self._bullet_client.changeVisualShape(self._body_id, i, rgbaColor=[0.8,0.8,0.8,1]) # Hand Color

        # define force and speed (movement of mount)
        self._force1 = 10000
        self._force2 = 500
        self._speed1 = 0.02
        self._speed2 = 0.5
        self.fix()


    def close(self, oid, flist):
        '''
            the post-process to generate stable grasp
        '''
        flist = np.ones(6)
        flist = flist[-5:]
        moving_joints = [
            [12, 14, 16], [20, 22, 24], [28, 30, 32], [36, 38, 40], [48, 50]
        ]
        velocities = [
            [0.8, 0.8, 0.6], [0.8, 0.8, 0.6], [0.8, 0.8, 0.6], [0.8, 0.8, 0.6], [1.2, 1.2]
        ]
        moving_joints = [moving_joints[i] for i in range(len(flist)) if flist[i]]
        velocities = [velocities[i] for i in range(len(flist)) if flist[i]]
        midxs = [0 for i in range(len(moving_joints))]
        close_joints = reduce(lambda a,b: a + b, moving_joints)
        contact_joints = []        
        jv_dict = dict(zip(reduce(lambda a,b: a + b, moving_joints), reduce(lambda a,b: a + b, velocities)))
        for i in range(60):
            close_joints = reduce(lambda a,b: a + b, [moving_joints[i][midxs[i]:] for i in range(len(moving_joints))])
            fix_joints = reduce(lambda a,b: a + b, [moving_joints[i][:midxs[i]] for i in range(len(moving_joints))])
            if len(close_joints) == 0:
                break
            for i in range(len(moving_joints)):
                for j in range(midxs[i],len(moving_joints[i])):
                    points = self._bullet_client.getContactPoints(self._body_id, oid, moving_joints[i][j]+1, -1)
                    if len(points) >= 1:
                        contact_joints.append(moving_joints[i][j])
                        midxs[i] = j+1
            self.fix_joints(fix_joints)
            self._bullet_client.setJointMotorControlArray(
                self._body_id,
                close_joints,
                self._bullet_client.VELOCITY_CONTROL,
                targetVelocities = [jv_dict[k] for k in close_joints],
                forces = [self._force2] * len(close_joints)
            )
            self.sleep_with_state_saving(oid, 1/120, 1)
        self.fix_joints(close_joints)
        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            contact_joints,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocities = [jv_dict[k] * 2 for k in contact_joints], # Train-2 | Test-3
            forces = [self._force2] * len(contact_joints),
        )
        self.sleep_with_state_saving(oid, 0.5, 11)

    def open(self):
        pass

    def lift(self):
        # move up the gripper to test whether the object can be lifted
        joint_ids = [0, 1, 2]
        current_states = np.array([self._bullet_client.getJointState(self._body_id, joint_id)[0] for joint_id in joint_ids])
        current_states[2] = current_states[2] + self.height
        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=current_states,
            forces=[self._force1] * len(joint_ids),
            positionGains=[self._speed1/4] * len(joint_ids)
        )
    
    def fix_joints(self, joint_ids):
        current_states = np.array([self._bullet_client.getJointState(self._body_id, joint_id)[0] for joint_id in joint_ids])
        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=current_states,
            forces=[self._force2] * len(joint_ids),
            positionGains=[self._speed2] * len(joint_ids),
        )

    def fix(self):
        finger_joints = range(7, self._bullet_client.getNumJoints(self._body_id))
        self.fix_joints(finger_joints)
    
    def setHeight(self, height):
        self.height = height

    def get_joint_states(self, dofs):
        return self._bullet_client.getJointStates(self._body_id, dofs)

    def get_hand_dof_values(self):
        dof_idxs = self.ex_dofs + self.hand_dofs
        dofs_data = self.get_joint_states(dof_idxs)
        s = np.array([dof[0] for dof in dofs_data])
        v = np.array([dof[1] for dof in dofs_data])
        s[3:6] = s[3:6][::-1]
        s[:3] *= 10
        v[3:6] = v[3:6][::-1]
        v[:3] *= 10
        return s,v
    
    def move(self, new_dofs, reset=False):
        dof_list = (self.ex_dofs + self.hand_dofs + self.undofs)
        speed_list = [0.2] * len(dof_list)
        force_list = [self._force2] * 24 + [self._force2 * 0.8] * 4
        ###
        hand_val = new_dofs[6:]
        ex_val = np.zeros(6)
        under_val = np.zeros(4)
        ex_val[:3] = new_dofs[:3]/10
        ex_val[3:6] = new_dofs[3:6][::-1]
        under_val[0] = new_dofs[10] * 0.8
        under_val[1] = new_dofs[12] * 0.8
        under_val[2] = new_dofs[15] * 0.8
        under_val[3] = new_dofs[18] * 0.8
        ###
        dofs_data = self._bullet_client.getJointStates(self._body_id, [3, 4, 5])
        if not reset:
            cur_dofs = np.array([dof[0] for dof in dofs_data])
            for i in [0,2]:
                while ex_val[3+i] < cur_dofs[i] - np.pi:
                    ex_val[3+i] += np.pi * 2
                while ex_val[3+i] > cur_dofs[i] + np.pi:
                    ex_val[3+i] -= np.pi * 2
            target_dofs = np.concatenate([ex_val, hand_val, under_val], axis=-1)
            self._bullet_client.setJointMotorControlArray(
                self._body_id,
                dof_list,
                self._bullet_client.POSITION_CONTROL,
                targetPositions = target_dofs,
                forces = force_list,
                positionGains = speed_list,
            )
            return target_dofs, dof_list
        else:
            target_dofs = np.concatenate([ex_val, hand_val, under_val], axis=-1)
            self._bullet_client.setRealTimeSimulation(0)
            for i in range(target_dofs.shape[0]):
                self._bullet_client.resetJointState(
                    self._body_id,
                    dof_list[i],
                    targetValue=target_dofs[i],
                    targetVelocity=0,
                )
            self._bullet_client.setJointMotorControlArray(
                self._body_id,
                dof_list,
                self._bullet_client.POSITION_CONTROL,
                targetPositions=target_dofs,
                forces=force_list,
                positionGains=speed_list
            )
            return target_dofs, dof_list


class SimpleSimulation(object):
    def __init__(self):
        self.reset_env()

    def reset_env(self):
        self.client = bc.BulletClient(connection_mode=p.GUI, options="--background_color_red=0 --background_color_blue=0 --background_color_green=0")
        self.client.resetDebugVisualizerCamera(0.8, 60, -40, (0,0,0.1))
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -10)
        self.tolerance = False
        self.type = None
        self.log_id = None
        self.object_mass = 0.1
        self.object_lateralFriction = 1.0
        # self.client.setTimeStep(1/1000) # If wang to reduce penetration in post-process, using this
        self.client.setRealTimeSimulation(0)
        # Load Plane 
        self.pid = self.client.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])
        self.oid = None # Object Holder
        self.client.changeDynamics(self.pid, -1, lateralFriction=0.6)
        # Load Hand
        self.mount_base_pose = np.array([0, 0, 1]) # [0, 0, 1m]
        self.mount_gripper = Gripper(self.client, self.mount_base_pose)
        self.bid = self.mount_gripper._body_id
        self.mount_gripper.move(np.zeros(24), reset=True)
        self.urdf_dirs = os.path.join(os.getcwd(), 'Grasp_Dataset_v4')


    def initalize_obj(self, obj_name, obj_config, type=0):
        '''
            Add the object to the simulator
        '''
        #  Transfer
        inital_pos = [obj_config[0]/10, obj_config[1]/10, obj_config[2]/10]
        inital_rot = [obj_config[4], obj_config[5], obj_config[6], obj_config[3]]

        # Reset the gripper pose
        self.mount_gripper.move(np.zeros(24), reset=True)
        
        # Remove the old object
        if self.oid is not None:
            self.client.removeBody(self.oid)
        shape_name = "%s_vhacd.urdf"%(obj_name)
        shape_urdf = os.path.join(self.urdf_dirs, obj_name, 'urdf', shape_name)
        if not os.path.exists(shape_urdf):
            print("Can't find the object")
            return None

        
        # Load the object into Pybullet
        self.type = type
        self.oid = self.client.loadURDF(shape_urdf, inital_pos, inital_rot, globalScaling=1)
        assert self.oid == 2
        if self.type == 0:
            # Hand can
            self.client.changeDynamics(self.oid, -1, lateralFriction=self.object_lateralFriction, spinningFriction=0.003, mass=self.object_mass)
        else:
            # Hand doesn't
            self.client.changeDynamics(self.oid, -1, lateralFriction=self.object_lateralFriction, spinningFriction=0.003, mass=0.0)
            self.collision_off()
        self.wait_till_stable()
        state = self.client.getBasePositionAndOrientation(self.oid)
        self.client.resetBasePositionAndOrientation(self.oid, [0, 0, state[0][2]], state[1])

        # Transfer
        new_state = np.array([0, 0, state[0][2] * 10, state[1][3], state[1][0], state[1][1], state[1][2]])
        return new_state
    
    def wait_till_stable(self, wait_iter=100):
        """
            Wait till the object is settled
        """
        last_state = self.client.getBasePositionAndOrientation(self.oid)
        for i in range(wait_iter):
            for i in range(24):
                self.client.stepSimulation()
            current_state = self.client.getBasePositionAndOrientation(self.oid)
            stable_flag = True
            max_diff_position = np.max(np.abs(np.array(last_state[0]) - np.array(current_state[0])))
            max_diff_orientation = np.max(np.abs(np.array(last_state[1]) - np.array(current_state[1])))
            if max_diff_position > 1e-5 or max_diff_orientation > 1e-4:
                stable_flag = False
            if stable_flag:
                break
            last_state = current_state

    def simulation_reduction(fn):
        '''
            the wrapper function to start simulation
        '''
        def wrapper_function(self, *args, **kwargs):
            self.client.setRealTimeSimulation(1)
            value = fn(self, *args, **kwargs)
            self.client.setRealTimeSimulation(0)
            return value
        return wrapper_function
    
    @ simulation_reduction
    def grasp_test(self, fstate, offset=True):
        next_state = fstate.copy()
        if offset:
            obj_offset = np.array(self.client.getBasePositionAndOrientation(self.oid)[0])
            next_state[:3] += obj_offset * 10
        next_state[:3] -= self.mount_base_pose * 10
        self.mount_gripper.move(next_state, reset=True)
        return True

    @ simulation_reduction
    def lift_test(self, flist,post=True):
        '''
            Test Whether the object can be lifted
        '''
        if post:
            # using post-process
            self.mount_gripper.close(self.oid, flist)
        
        # Lift the gripper
        info1 = np.array(self.client.getBasePositionAndOrientation(self.oid)[0])
        self.mount_gripper.setHeight(0.6)
        self.mount_gripper.lift()
        self.mount_gripper.sleep_with_state_saving(self.oid, 1.0, 2)

        # Waiting until object is static
        p1 = None
        p2 = None
        counter = 0
        for i in range(240):
            p2 = np.array(self.client.getBasePositionAndOrientation(self.oid)[0])
            if p1 is None:
                p1 = p2
                continue
            if np.min(p2-p1) < 1e-4:
                counter += 1
            if counter >= 10:
                break
            p1 = p2
            self.mount_gripper.sleep_with_state_saving(self.oid, 1/120, 2)
        info2 = p2

        # Check whether the center of object is lifted more than given value
        pos_change = (info2 - info1) 
        if pos_change[2] > 0.2:
            return True
        else:
            return False

    @ simulation_reduction
    def hand_move(self, hstate, reset=False):
        next_state = hstate.copy()
        next_state[:3] -= self.mount_base_pose * 10
        tp, dl = self.mount_gripper.move(next_state, reset=reset)
        if not reset:
            # dynamic simulation
            self.mount_gripper.sleep_with_state_saving(self.oid, 5/120, 0)
        else: 
            # kinematic simulation
            self.mount_gripper.sleep_with_state_saving(self.oid, 1/120, 0)
        s,v = self.get_hand_dof_values() # The DOF values of the gripper (position & velocity)
        pose = self.get_obj_pose_values() # The pose of the object 
        return s, v, pose
    
    def hand_reset(self, state):
        new_state = state.copy()
        new_state[:3] -= (self.mount_base_pose * 10) # Base offset
        self.mount_gripper.move(new_state, reset=True)

    def obj_reset(self, obj_config):
        pos = [obj_config[0]/10, obj_config[1]/10, obj_config[2]/10]
        rot = [obj_config[4], obj_config[5], obj_config[6], obj_config[3]]
        self.client.resetBasePositionAndOrientation(self.oid, pos, rot)

    def get_obj_pose_values(self):
        pose = self.client.getBasePositionAndOrientation(self.oid)
        new_pose = np.zeros(7)
        new_pose[:3] = pose[0]
        new_pose[:3] *= 10
        new_pose[4:] = pose[1][:3]
        new_pose[3] = pose[1][3]
        return new_pose
    
    def get_hand_dof_values(self):
        s, v = self.mount_gripper.get_hand_dof_values()
        s[:3] += self.mount_base_pose * 10
        return s, v

    def collision_off(self): 
        assert self.oid is not None
        self.tolerance = True
        for i in range(-1, self.client.getNumJoints(self.bid)):
            self.client.setCollisionFilterPair(self.bid, self.oid, i, -1, 0)
            self.client.setCollisionFilterPair(self.bid, self.pid, i, -1, 0)

    def collision_on(self): 
        assert self.oid is not None
        self.tolerance = False
        for i in range(-1, self.client.getNumJoints(self.bid)):
            self.client.setCollisionFilterPair(self.bid, self.oid, i, -1, 1)
            self.client.setCollisionFilterPair(self.bid, self.pid, i, -1, 1)

    def start_logging_video(self, url):
        self.log_id = self.client.startStateLogging(self.client.STATE_LOGGING_VIDEO_MP4, url)

    def stop_logging_video(self):
        if self.log_id is not None:
            self.client.stopStateLogging(self.log_id)
            self.log_id = None

    def recover_mass(self):
        self.client.changeDynamics(self.oid, -1, mass=self.object_mass)

    def get_state_data(self):
        return self.mount_gripper.get_state_data()
    

            




