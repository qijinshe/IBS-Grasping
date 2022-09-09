#!/usr/bin/env python
import os
import time
from scipy.spatial.transform import Rotation as R
from numpy import random
import numpy as np
import open3d as o3d
# ROS
from ibs_env.srv import *
from ibs_env.msg import *
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import rospy
# Code
from hand import Hand
from utility import xyz_array_to_pointcloud2, xyzl_array_to_pointcloud2
from hand_utils import config_from_xml
from manipulator_euler import HandManipulator
from hand_file import SimpleSimulation
# Grasp
from generalizedQ1 import compute_generalized_q1, compute_pentration
#
import gym


random.seed(123456)
point_num = 4096 # IBS point num
# ROS Service Name

ibs_service_name = "get_ibs6"
load_service_name = "load_obj6"
load_fp_service_name = "load_obj_fp"


def get_ibs(hand_points, center):
    '''
        Call the IBS calculation module
    '''
    rospy.wait_for_service(ibs_service_name)
    msg = xyz_array_to_pointcloud2(hand_points)
    try:
        func = rospy.ServiceProxy(ibs_service_name, GetIBSHand2)
        res = func(msg, center[0], center[1], center[2])
        # Compute the IBS points
        ibs_points = np.array(list(point_cloud2.read_points(res.ibs, 
            field_names=("x", "y", "z","nor1x","nor1y","nor1z","nor2x","nor2y","nor2z","dis1","dis2","dis3"),skip_nans=True)))

        # Near-contact points on object
        oc = np.array(list(point_cloud2.read_points(res.ocontact, 
            field_names=("x", "y", "z","normal_x","normal_y","normal_z","idx_h","idx_i"),skip_nans=True)))
        if oc.shape[0] > 0:
            op = oc[:,:6]
            hi = oc[:,6].astype('int32') # the index of corresponding hand points 
            ii = oc[:,7].astype('int32') # the_index of IBS points
            ip = ibs_points[ii] # the semi-contact IBS points
        else:
            # There is no near-contact points on the IBS
            hi = np.array([]).astype('int32')
            ii = np.array([]).astype('int32')
            op = np.array([]).astype('int32')
            ip = np.array([]).astype('int32')
    
        # Separate information of "dis3" (dis3 = env_label * (idx_on_hand+100))
        env_label = (ibs_points[:,-1] >= 100)
        idx_on_hand = (np.abs(ibs_points[:,-1])-100)
        ibs_points[:,-1] = idx_on_hand
        ip = None
        return (ibs_points, op, ip, hi, ii, env_label)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


def load_obj_with_pose(name, ox, oy, oz, w, x, y, z):
    '''
        Load an object to the environment (url & pose)
    '''
    rospy.wait_for_service(load_service_name)
    try:
        func = rospy.ServiceProxy(load_service_name, LoadObj2)
        res = func(name, w, x, y, z, ox, oy, oz)
        return res.ok
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


def load_pointcloud_as_obj(env_points, center):
    '''
        Load an object to the environment (pointcloud)
    '''
    msg = xyzl_array_to_pointcloud2(env_points)
    rospy.wait_for_service(load_fp_service_name)
    try:
        func = rospy.ServiceProxy(load_fp_service_name, LoadObjFP)
        res = func(msg, center[0], center[1], center[2])
        return res.ok
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


class IBSEnv(gym.Env):
    '''
        IBS Grasp Environment
    '''
    def __init__(self, simulation = False, feedback=False):
        # self.resoluation = 600
        self.resoluation = 200
        self.cur_mode = None
        self.record = False
        self.record_dir = 'videos'
        self.feedback = feedback
        self.simulation = simulation
        self.update_obj = True
        self.reset_move = True
        # Publish useful information
        self.point_pub = rospy.Publisher('Hand_Core', PointCloud2, queue_size=10)
        self.pub1 = rospy.Publisher('normal_contact', finger_info, queue_size=10)
        self.pub2 = rospy.Publisher('contact', finger_info, queue_size=10)
        self.pub3 = rospy.Publisher('collision', finger_info, queue_size=10)
        self.pub4 = rospy.Publisher('reward', finger_info, queue_size=10)
        if simulation:
            self.real_env = SimpleSimulation()
        else:
            self.real_env = None
        # Hand
        path = "src/ibs_env/scripts/hand/ShadowHand/"
        dofs = 24
        self.manipulator = HandManipulator(path, dofs)
        self.lb = self.manipulator.joints_LB
        self.ub = self.manipulator.joints_UB
        # Object
        self.opose = None
        self.tar_dor = None
        self.state_dim = (point_num * (11+5+2) + self.manipulator.action_dim)
        self.action_space = self.manipulator.action_space
        # Counters
        self.counter_train = -1
        self.counter_test = -1
        self.counter_eval = -1
        self.counter_demo = -1
        self.tasks = None
        self.tasks_test = None
        ###
        self.cur_shape = None
        self.cur_init = None
        self.cur_mode = -1
        self.pens = None
        # Object dataset
        self.data_dir = os.path.join(os.getcwd(), 'Grasp_Dataset_v3')
        self.shapes_train = [shape for shape in os.listdir(self.data_dir) if shape[:2] == 'gd' or shape[:3] == 'kit'] # Training Set
        self.shapes_eval = [shape for shape in os.listdir(self.data_dir) if shape[:3] == 'ycb'] # Validation Set
        self.shapes_test = [shape for shape in os.listdir(self.data_dir) if shape[:3] == 'ycb'] # Testing Set
        # self.shapes_test = [shape for shape in os.listdir(self.data_dir) if shape[:7] == 'bigbird' or shape[:3] == 'ycb'] # Testing Set
        ###
        self.shapes_custom = []
        self.view_set = []
        ###
        if not self.read_file():
            self.init_taskset()

    def get_state_data(self): # !!!Name
        if self.real_env is not None:
            return self.real_env.get_state_data()
    
    def init_taskset(self):
        total_num = 20
        task_train = []
        ccc = 0
        ttt = len(self.shapes_train)
        for shape in self.shapes_train:
            ccc += 1
            print("%d/%d"%(ccc,ttt))
            shape_url = os.path.join(self.data_dir, shape, 'pcd', '600.pcd' )
            grasps_url = os.path.join(self.data_dir, shape, 'grasps')
            scan = 0
            add = 0
            tmp = []
            while True:
                scan += 1
                cur_grasp = os.path.join(grasps_url, "grasp_%d.xml"%(scan))
                if (add >= total_num) or (not os.path.exists(cur_grasp)):
                    task_train += tmp
                    if len(tmp) == 0:
                        print("No feasible grasp for the object %s"%(shape))
                    self.tasks = task_train
                    self.write_file()
                    break
                flag, q1 = self.check_pos(shape_url, cur_grasp)
                if flag:
                    tmp.append((shape_url, cur_grasp, q1))
                    add += 1
        self.tasks = task_train
        self.write_file()
        print("Total Demo Number:\t", len(task_train))

    def get_current_ibs(self):
        sampled = self.manipulator.getPointCloud()
        ibs, op, ip, hi, ii, el = get_ibs(sampled, self.manipulator.hand_center)
        self.normal_labels = self.get_normal_label(ibs)
        self.env_labels = el
        return (ibs, op, ip, hi, ii, el)

    def get_normal_label(self,ibs):
        return self.manipulator.get_internal_label(ibs[:,-1])
    
    def check_pos(self, shape_url, grasp_url):
        hand_config, obj_config = config_from_xml(self.manipulator.hand, grasp_url)
        self.opose = obj_config
        cur_dof = hand_config
        load_obj_with_pose(shape_url, *self.opose)
        sampled = self.manipulator.reset(cur_dof)
        ibs, op, ip, hi, ii, el = self.get_current_ibs()
        _, obs = self.fix_ibs(ibs)
        ip = ibs[ii]
        ibs[:,-1] = self.manipulator.get_label(ibs[:,-1])

        flag_cls_env = self.get_collision_by_ibs(op[~el[ii]], ip[~el[ii]])
        flag_floor = flag_cls_env[0] > 3

        flag_out = (ibs.shape[0] < 1)
        flag_pos = (hand_config[2] <= obj_config[2])
        if flag_floor or flag_out or flag_pos:
            return False, 0
        shape = grasp_url.split('/')[-3]
        self.cur_shape = shape
        q1 = compute_generalized_q1(self.manipulator.hand, shape, hand_config, 4, obj_config=self.opose)
        if q1 < 0.04:
            return False, q1
        print(shape)
        print("Q1:\t", q1)
        return True, q1

    def write_file(self):
        print("Write Grasp to File!")
        file_object = open('demo_list.txt', "w")
        for task in self.tasks:
            file_object.write("%s %s %.4f\n"%(task[0], task[1], task[2]))
        file_object.close()

    def read_file(self):
        if not os.path.exists('demo_list.txt'):
            return False
        tasks = []
        file_object = open('demo_list.txt', "r")
        for line in file_object.readlines():
            task = tuple(line.split())
            tasks.append(task)
        file_object.close()
        self.tasks = tasks
        print("Train:  ", len(self.tasks))
        return True

    def reset_counter(self):
        # self.counter_train = -1
        self.counter_eval = -1

    def real_obj_config(self, obj_config=None, type=0):
        if self.real_env is None:
            if obj_config is None:
                new_state = np.array([0,0,0,1,0,0,0])
            else:
                new_state = obj_config
        else:
            if obj_config is None:
                new_state = self.real_env.initalize_obj(self.cur_shape, [0,0,0.8,1,0,0,0], type=type)
            else:
                new_state = self.real_env.initalize_obj(self.cur_shape, obj_config, type=type)
        return new_state

    def execute(self, action):
        if self.feedback and self.real_env is not None:
            # dynamic
            self.execute_with_feedback(action)
        else:
            # kinematic
            self.manipulator.execute(action)

    def update_env(self, mode): 
        self.cur_mode = mode
        # mode:
        angle_list = None
        def_rad = 2
        if mode in [0, 2, 3]:
            if mode == 0: # Training
                self.counter_train += 1
                shape_id = self.counter_train
                shape = self.shapes_train[shape_id%len(self.shapes_train)]
            elif mode == 2: # Evaluation
                angle_list = [[0, 0, 0.125],[0.5, 0, 0.125],[1, 0, 0.125]]
                self.counter_eval += 1
                shape_id = (self.counter_eval // len(angle_list))
                shape = self.shapes_eval[shape_id%len(self.shapes_eval)]
            elif mode == 3:  # Test
                angle_list = [[0,0,0.125*i] for i in range(8)] + [[0.50,0,0.125*i] for i in range(8)] + [[0.99,0,0.125*i] for i in range(8)]
                self.counter_test += 1
                shape_id = (self.counter_test // len(angle_list))
                shape = self.shapes_test[shape_id%len(self.shapes_test)]
            #
            print("Shape:\t", shape)
            self.cur_shape = shape
            shape_url = os.path.join(self.data_dir, shape, 'pcd', '%d.pcd'%(self.resoluation))
            obj_config = self.real_obj_config() # Drop
            cur_dof = np.zeros(self.manipulator.action_dim) # 24
            # Initial Configuration
            if mode == 0:
                angle = np.random.rand(3)
                self.cur_init = "%.3f-%.3f-%.3f"%(angle[0], angle[1], angle[2])
                rad = def_rad * (0.95 + np.random.rand(1) * 0.1)
                joint_disturb = np.clip(np.random.randn(self.manipulator.action_dim-6)/10,-1,1)
                cur_dof[6:] = np.max(joint_disturb,0) * self.ub - np.min(joint_disturb,0) * self.lb
            elif mode == 2:
                angle = angle_list[self.counter_eval%len(angle_list)]
                self.cur_init = "%.3f-%.3f-%.3f"%(angle[0], angle[1], angle[2])
                rad = def_rad
                cur_dof[6:] = 0
            elif mode == 3:
                angle = angle_list[self.counter_test%len(angle_list)]
                self.cur_init = "%.3f-%.3f-%.3f"%(angle[0], angle[1], angle[2])
                rad = def_rad
                cur_dof[6:] = 0
            roll = -(0.5+angle[0]/2) * np.pi
            yaw =  -(angle[1]/6) * np.pi
            pitch = np.pi * (2*angle[2]-1)
            euler = np.array([roll, yaw, pitch])
            ##########################
            cur_dof[:3] = [-rad * np.sin(pitch) * np.sin(roll), rad * np.cos(pitch) * np.sin(roll), 0.1-rad * np.cos(roll)]
            cur_dof[:3] += obj_config[:3]
            cur_dof[:3] -= self.manipulator.get_offset(euler)
            cur_dof[3:6] = euler

        if mode in [1,4]:
            self.cur_mode = mode
            self.counter_demo += 1
            hand_config = None
            obj_config = None
            print(len(self.tasks))
            if hand_config is None:
                shape_url, grasp_url, val  = self.tasks[self.counter_demo%(len(self.tasks))]
                hand_config, obj_config = config_from_xml(self.manipulator.hand, grasp_url)
            print("Shape:\t", grasp_url.split('/')[-3])
            print("Grasp:\t", grasp_url.split('/')[-1])
            print("Data Value:\t", val)
            self.cur_init =  grasp_url.split('/')[-1]
            self.cur_shape = grasp_url.split('/')[-3]
            points = np.asarray(o3d.io.read_point_cloud(shape_url).points)
            min_z = np.min(points[:,2])
            obj_config2 = self.real_obj_config(obj_config=np.array([0, 0, -min_z, *obj_config[3:]]), type=1)
            hand_config[:3] += (obj_config2[:3]-obj_config[:3])
            obj_config = obj_config2
            self.tar_dof = hand_config.copy()
            cur_dof = hand_config.copy() #25
            if mode == 1:
                offset = self.manipulator.get_offset(hand_config[3:7])
                vec = (hand_config[:3] - offset - obj_config[:3])
                vec = vec / np.sqrt(np.sum(vec**2))
                cur_dof[:3] = obj_config[:3] + (2.0 * vec)
                joint_disturb = np.clip(np.random.randn(self.manipulator.action_dim-6)/10,-1,1)
                cur_dof[7:] = np.max(joint_disturb,0) * self.ub - np.min(joint_disturb,0) * self.lb
                cur_dof[7:] = 0
        # Object Reset
        print("Object Config:\t", obj_config)
        self.opose = obj_config
        load_obj_with_pose(shape_url, *self.opose)
        self.cur_shape_url = shape_url
        # Manipulator Reset
        self.manipulator.reset(cur_dof)


    def reset(self, mode=0 ,obj=None, euler=None):
        self.TTL = 0
        self.coll = 0
        self.pens = []
        self.tar_dof = None
        self.cur_init = None # Name
        self.cur_mode = mode
        self.cur_shape_url = None
        self.hold_on = 0
        self.update_env(mode)

        # record process if using feedback control
        if self.record and self.feedback:
            self.real_env.stop_logging_video()
            dir = os.path.join(self.record_dir, self.cur_shape, self.cur_init)
            if not os.path.exists(dir):
                os.makedirs(dir)
            url = os.path.join(dir, 'process.mp4')
            self.real_env.start_logging_video(url)

        # Observation Calculation
        ibs, op, ip, _, ii, el = self.get_current_ibs()
        _, obs = self.fix_ibs(ibs)
        ip = ibs[ii]
        self.get_collision_by_ibs(op, ip)
        if self.feedback:
            self.real_env.hand_reset(self.manipulator.get_euler_state())
        return obs


    def step(self, action, stop):
        self.TTL += 1
        pens = []
        kinds_c = np.zeros(6) # ibs contact point number of different gripper components
        kinds_bc = np.zeros(6) # ibs contact point one the inside of the plam number of different gripper componets
        reward = np.zeros(7) # Reward Vector: [grasp reward, reaching reward of p0, ..., reaching reward of p5]
        done = np.zeros(7)
        success = False
        q1 = 0
        self.execute(action) # Forward_kinematics
        # Get Current IBS
        ibs, op, ip, hi, ii, el = self.get_current_ibs()        
        _, obs = self.fix_ibs(ibs)
        for i in ibs[:,-1][ii][el[ii]]:
            kinds_c[int(i)] += 1
        for i in range(6):
            mask = (ibs[:,-1][ii][el[ii]]==i)
            kinds_bc[i] = np.sum(self.normal_labels[ii][el[ii]][mask]>0)
        ip = ibs[ii]
        flag_cls = self.get_collision_by_ibs(op, ip) # collision point number between the IBS and the object
        # flag_cls_env = self.get_collision_by_ibs(op[~el[ii]], ip[~el[ii]])
        coll_flag = np.array([flag_cls[i+1] for i in range(6)]) > 3
        self.coll += bool(max(coll_flag))
        # Stop Check
        flist = np.array([int(kinds_c[i]>3) for i in range(6)])
        fins = np.sum(flist)
        if fins >= 2: # at least two gripper parts 
            rd = np.random.rand()
            if rd < stop:
                quat_state = self.manipulator.get_quat_state()
                q1 = self.get_generalized_q1(self.cur_shape, quat_state, 4, pens)
                if self.real_env is not None:
                    success = self.simulation_test(self.manipulator.get_euler_state(), flist)
                    print("Success Flag:", success)
                    reward[0] = (q1 * 1000 + success * 150) - 1
                else:
                    reward[0] = (q1 * 1000) - 1
                done[0] = 1
            self.hold_on += fins
        else:
            self.hold_on = 0

        flag_out = (ibs.shape[0] <= 1)
        if flag_out: 
            # NO IBS points in the sampling range
            print("IBS FAIL") 
        
        # Calculate rewards
        reward[0] = (reward[0] - 3)
        for i in range(6):
            if kinds_bc[i] > 3:
                reward[i+1] += min(40, kinds_bc[i]) * 0.25
            all_coll = flag_cls[i+1]
            if all_coll > 3:
                reward[i+1] = -100 * 0.25
            done[i+1] = 1

        # Publish infomation to ROS
        self.pub1.publish(finger_info(kinds_bc))
        self.pub2.publish(finger_info(kinds_c))
        self.pub3.publish(finger_info(flag_cls[1:]))
        self.pub4.publish(finger_info(reward))
        valid_fin_num = np.sum(np.array([kinds_bc[i]>3 for i in range(6)]))
        pens.append(0)
        info = dict()
        info["Q1"] = (q1 * 100)
        info['Fingers'] = fins
        info['VFingers'] = valid_fin_num
        info["Collis"] = self.coll
        info['Len'] = self.TTL
        info['FinPen'] = pens[0]
        info["SuccessGrasp"] = success
        return (obs, reward, done, info)

    def get_generalized_q1(self, shape, grasp, dir_num, pens=[]):
        q1 = compute_generalized_q1(self.manipulator.hand, shape, grasp, dir_num, pens=pens, use_center=False, obj_config=self.opose)
        return q1

    def setRecorder(self, flag):
        if isinstance(flag, bool) and flag:
            print("record the grasp")
            self.record = True

        if isinstance(flag, str):
            print("record the grasp")
            print("video directory change to %s"%(flag))
            self.record = True
            self.record_dir = flag

    def setResetMove(self, reset_move):
        if self.feedback:
            self.reset_move = reset_move

    def setUpdateObj(self, flag):
        assert isinstance(flag, bool)
        if self.feedback:
            self.update_obj = flag

    def simulation_test(self, grasp, flist):
        if self.real_env.type == 0:
            if not self.feedback:
                self.real_env.grasp_test(grasp, offset=False)
            if self.record:
                self.real_env.stop_logging_video()
                dir = os.path.join(self.record_dir, self.cur_shape, self.cur_init)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                url = os.path.join(dir, 'lift.mp4')
                self.real_env.start_logging_video(url)
            success = self.real_env.lift_test(flist)
            if self.record:
                self.real_env.stop_logging_video()
                with open(dir+"/"+str(success)+".txt", 'w') as f:
                    f.write(str(success))
        elif self.real_env.type in [1,2]: # tolerant
            if self.real_env.type == 1:
                self.real_env.recover_mass()
            self.real_env.collision_on()
            if not self.feedback:
                self.real_env.grasp_test(grasp, offset=False)
            success = self.real_env.lift_test(flist)
        else:
            assert False
        return success

    def execute_with_feedback(self, action):
        assert self.real_env is not None
        new_state = self.manipulator.bound_state(action)
        s, _, p = self.real_env.hand_move(new_state, self.reset_move)
        self.opose = p
        self.manipulator.reset(s)
        if self.update_obj:
            load_obj_with_pose('', *p)

    def fix_ibs(self, ibs, num=point_num):
        ibs[:,-1] = self.manipulator.get_label(ibs[:,-1])
        new_ibs = None
        # Sample IBS points to given number
        if ibs.shape[0] >= num:
            idxs = [i for i in range(ibs.shape[0])]
            idxs = list(np.random.choice(idxs, num, replace=False))
            new_ibs = ibs[idxs]
            nor_lab = self.normal_labels[idxs]
            env_lab = self.env_labels[idxs]
        else:
            idxs = [i for i in range(len(ibs))]
            res = list(np.random.choice(idxs, num-len(ibs), replace=True))
            new_ibs = ibs[(idxs+res)]
            nor_lab = self.normal_labels[(idxs+res)]
            env_lab = self.env_labels[(idxs+res)]
        # Keep 
        self.ibs = new_ibs.copy()
        self.ibs[:,-1] = (self.ibs[:,-1]+1) * env_lab
        # Modified IBS data
        new_ibs = np.concatenate((new_ibs[:,:-1], nor_lab.reshape((-1,1)), env_lab.reshape((-1,1)), np.eye(6,5,k=-1)[new_ibs[:,-1].astype('int32')]), axis=-1) # !!!
        new_ibs[:,:3] = (new_ibs[:,:3] - self.manipulator.hand_center) # 0424
        new_ibs = new_ibs.transpose((1,0))
        # Get Observation
        config = self.manipulator.get_state()
        obs = np.hstack([new_ibs.reshape((-1)), config]).astype('float32')
        return new_ibs, obs

    def get_collision_by_ibs(self, obj_p, ibs_p):
        flag_c = np.zeros(7)
        if obj_p.shape[0] > 0:
            cos = np.sum(obj_p[:,3:6] * ibs_p[:,6:9], axis=-1) > 0.5
            flag_c[0] = np.sum(cos)
            for label in ibs_p[cos,-1]:
                flag_c[int(label)+1] += 1  
        return flag_c
    
    def random_action(self):
        return self.manipulator.random_action()

    def hacker_action(self):
        action = self.manipulator.hacker_action(self.tar_dof)
        if np.sum(np.abs(action)) < 0.01:
            stop = 1
        else:
            stop = -1
        return action, stop


if __name__ == "__main__":
    rospy.init_node('ibs_publiser', anonymous=True)
    pub = rospy.Publisher('sampled_ibs', PointCloud2, queue_size=10)
    env = IBSEnv(simulation = True)
    counter = 0
    step = 0
    episode_reward = 0
    episode_num = 0
    ibs_data = []
    mode = 1
    env.reset(mode=mode) # ?
    while not rospy.is_shutdown():
        counter += 1
        action, stop = env.hacker_action()
        step += 1
        ibs_data.append(env.ibs)
        if env.ibs.shape[0] > 0:
            pc2 = xyzl_array_to_pointcloud2(np.concatenate([env.ibs[:,:3],env.ibs[:,-1:]],axis=-1), frame_id='object')
            pub.publish(pc2)
        start = time.time()
        obs, reward, done, info = env.step(action, stop)
        end = time.time()
        print("Time:", end-start)
        episode_reward += reward
        if done[0] or step == 200:
            print("Episode Reward: ", episode_reward)
            print("Q1:\t", info["Q1"])
            print("Coll:\t", info["Collis"])
            print("Fins:\t", info['Fingers'])
            counters = 0
            episode_reward = 0
            episode_num += 1
            if episode_num >= 2000:
                break
            step = 0
            env.reset(mode=mode)
            time.sleep(1)



    
    