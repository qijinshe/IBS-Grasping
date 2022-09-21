#!/usr/bin/env python

import argparse
import itertools
from re import T
from tokenize import Name
import numpy as np
import matplotlib.pyplot as plt
import time
import rospy
import os
import torch
import pandas as pd

from sensor_msgs.msg import PointCloud2

# from utility import Train_Type
from utility import xyzl_array_to_pointcloud2
from grasping import IBSEnv
from sac import SAC_FIN_STOP
from sac import ReplayMemory

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',  #R
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G', #
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G', #
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.02, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--start_steps', type=int, default=50000, metavar='Nastype', # change
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--replay_size', type=int, default=30000, metavar='N',
                    help='size of replay buffer (default: 1000000)')

parser.add_argument('--model_name', type=str, default='fine', metavar='NAME',
                    help="the name of the model (default: fine")
parser.add_argument('--quick', action='store_true', help="test on small cases")
parser.add_argument('--train_model', action='store_true', help="train model")
parser.add_argument('--con', action='store_true', help="continue train the given model")
args = parser.parse_args()

WORD = "remove"
DATE = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime())
NAME = DATE + "_" + WORD # Model Name
TEST_PREFIX = "" # Test file prefix

MAX_EP_STEPS = 200 # Max Eposide Steps
TRAIN_MODEL = args.train_model  # Train model
CONTINUE = args.con # Continue Training the given model

SIMULATION = True # Simulate the grasp in Pybullet
RECORD = False # Record the process (False/URL)

FEEDBACK = True # Simulated the whole process in Pybullet
HACKER = True # Jump to the given state / move by PD controller
UPDATE_OBJ = True # Whether update the object position

first_output = ""
if TRAIN_MODEL:
    first_output = "Train model %s"%(NAME)
    if CONTINUE:
        first_output = "Continue train model %s"%(NAME)
else:
    first_output = "Test model %s"%(NAME) 
print("Task: ", first_output)
time.sleep(3)

ememory = ReplayMemory(args.replay_size)
# dmemory = ReplayMemory(args.start_steps) #
dmemory = ReplayMemory(50000) #
pos_num = None

# TIMESTEP
updates = 0
success_times = 0
if TRAIN_MODEL:
    writer = SummaryWriter('M01/sac_' + NAME)
if CONTINUE:
    args.start_steps = 0

env = IBSEnv(simulation=SIMULATION, feedback=FEEDBACK)
s_dim = env.state_dim
a_dim = env.action_space
rl = SAC_FIN_STOP(s_dim, a_dim, args)


# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_num_threads(4)


def train_network():
    global updates
    global success_times
    counter1 = ememory.counter
    counter2 = dmemory.counter
    bound = counter1/(counter1+counter2+0.01)
    mem = None
    if np.random.rand() > bound or counter1 < args.batch_size:
        mem = dmemory
    else:
        mem = ememory
    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, dis_loss, n1, n2, dis1, dis2, diff = rl.update_parameters(mem, args.batch_size, updates)
    updates += 1
    # Tensorboard
    writer.add_scalar('Loss/Q1_loss', critic_1_loss, global_step=updates)
    writer.add_scalar('Loss/Q2_loss', critic_2_loss, global_step=updates)
    writer.add_scalar('Loss/policy_loss', policy_loss, global_step=updates)
    writer.add_scalar('Loss/ent_loss', ent_loss, global_step=updates)
    writer.add_scalar('Loss/alpha', alpha, global_step=updates)
    if (rl.self_counter-1) % 10 == 0:
        writer.add_scalar('Loss/selfCollision', (dis_loss/100), global_step=updates)
    

def train():
    rospy.init_node('ibs_publiser', anonymous=True)
    pub = rospy.Publisher('sampled_ibs', PointCloud2, queue_size=10)
    # Training Loop
    global env
    global updates
    global success_times
    global pos_num
    #
    pos_num = len(env.shapes_train * 1)
    eval_num = (len(env.shapes_eval) * 3)
    print("Task Num: ", pos_num)
    print("Test Num: ", eval_num)
    updates = 0
    success_times = 0
    total_numsteps = 1
    test_episode = 0
    for i_episode in itertools.count(1):
        if total_numsteps <= args.start_steps:
            state = env.reset(mode=1)
        else:
            state = env.reset(mode=0)
        episode_reward = 0
        episode_steps = 0
        expert_flag = False
        info = None
        done = False
        for i in range(1, MAX_EP_STEPS+1):
            if total_numsteps < args.start_steps:
                expert_flag = True
                action, stop = env.hacker_action()
            else: 
                action, stop = rl.select_action(state) # Sample action from policy
                if (ememory.counter + dmemory.counter) > args.batch_size:
                    for j in range(args.updates_per_step): # Number of updates per step in environment
                        start = time.time()
                        train_network() # Update parameters of all the networks
                        end = time.time()
                        print("Time: ", end-start, "\tStop:", stop)
            if env.ibs.shape[0] > 0:
                pc2 = xyzl_array_to_pointcloud2(np.concatenate([env.ibs[:,:3],env.ibs[:,-1:]],axis=-1), frame_id='object')
                pub.publish(pc2)
            next_state, reward, done, info = env.step(action, stop) # Step

            mask = (1 - np.array(done))
            episode_steps += 1
            total_numsteps += 1
            episode_reward += float(np.sum(reward))
            if not expert_flag:
                action_stop = np.zeros(25)
                action_stop[:24] = action
                action_stop[24] = stop
                ememory.push(state, action_stop, reward, next_state, mask) # Append transition to ememory
            else:
                if True:
                    action_stop = np.zeros(25)
                    action_stop[:24] = action
                    action_stop[24] = stop
                    dmemory.push(state, action_stop, reward, next_state, mask) # Append transition to ememory
            state = next_state
            if done[0]:
                break
        print("Episode: %d , Step: %d, Reward: %.2f, Fingers: %d, Q1, %.2f, Collision: %.2f"%(i_episode, total_numsteps, episode_reward, info['Fingers'], info['Q1'], info['Collis'])) # Print Infomation
        writer.add_scalar('Detail/EpisodeReward', episode_reward, global_step=i_episode)
        writer.add_scalar('Detail/Collision', info['Collis'], global_step=i_episode) # Collision
        writer.add_scalar('Detail/Fingers', info['Fingers'], global_step=i_episode) # Fingers Number
        writer.add_scalar('Detail/Length', info['Len'], global_step=i_episode) # Collision
        writer.add_scalar('Detail/Q1', info['Q1'], global_step=i_episode) # Q1 Metric
        if SIMULATION:
            writer.add_scalar('Detail/SuccessGrasp', info['SuccessGrasp'], global_step=i_episode) # Collision
        if total_numsteps > (args.num_steps + args.start_steps): 
            # Training ends
            break 
        if (total_numsteps > args.start_steps) and (i_episode % 20 == 0):
            # Save Trained Models
            rl.save_model("ibs", NAME)
            print("Model Saved!")
        if ((env.counter_train+env.counter_demo+2) % 1000 == 0) and (args.eval is True): 
            # Evalute the model
            test_episode += 1
            avg_reward = 0
            mean_q = 0
            mean_f = 0
            mean_c = 0
            mean_s = 0
            for i in range(eval_num):
                state = env.reset(mode=2)
                episode_reward = 0
                done = False
                for j in range(1, MAX_EP_STEPS+1):
                    action, stop = rl.select_action(state, evaluate=True)
                    next_state, reward, done, info = env.step(action, stop)
                    episode_reward += np.sum(reward)
                    state = next_state
                    if done[0]:
                        break
                avg_reward += episode_reward
                mean_q = (mean_q + info['Q1'])
                mean_f = (mean_f + info['Fingers'])
                mean_c = (mean_c + info['Collis'])
                mean_s = (mean_s + info["SuccessGrasp"])
            mean_q = (mean_q / eval_num)
            mean_f = (mean_f / eval_num)
            mean_c = (mean_c / eval_num)
            mean_s = (mean_s / eval_num)
            avg_reward = (avg_reward / eval_num)
            env.reset_counter()
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(test_episode, round(avg_reward, 2)))
            print("----------------------------------------")
            writer.add_scalar('Reward/MeanQ1', mean_q, global_step=i_episode)
            writer.add_scalar('Reward/MeanFingers', mean_f, global_step=i_episode)
            writer.add_scalar('Reward/MeanCollision', mean_c, global_step=i_episode)
            writer.add_scalar('Reward/MeanSuccessRate', mean_s, global_step=i_episode) # 0425
            writer.add_scalar('Reward/Evaluation', avg_reward, global_step=i_episode)
    env.close()


def eval(quick_check=False):
    rospy.init_node('ibs_publiser', anonymous=True)
    start =time.time()
    env.setRecorder(RECORD)
    env.setResetMove(HACKER)
    env.setUpdateObj(UPDATE_OBJ)
    all_step = 200
    if not HACKER:
        all_step = 500
    episode_reward = 0
    step = 0
    times = 0
    # statics
    shape_list = []
    grasp_list = []
    Q1_list = []
    Coll_list = []
    Fnum_list = []
    fp_list = list()
    success_list = list()
    start_pos_list = list()
    end_pos_list = list()
    start_obj_list = list()
    end_obj_list = list()
    ###
    mean_q = 0
    mean_f = 0
    mean_c = 0
    mean_s = 0
    mean_sq = 0
    mean_pp = 0.
    all_p = 0
    all_s = 0
    ###
    o_list = list()
    h_list = list()
    ibs_list = list()
    if not quick_check:
        test_num = (len(env.shapes_test) * 24)
        mode = 3
        print("Test Case: ", len(env.shapes_test))
    else:
        test_num = (len(env.shapes_eval) * 3)
        mode = 2
        print("Test Case: ", len(env.shapes_eval))
    state = env.reset(mode=mode)
    start_all = time.time()
    start = time.time()
    cur_counter = 0
    while cur_counter < test_num:
        config_o = env.opose.copy()
        config_h = env.manipulator.get_euler_state()
        if step == 0:
            start_pos_list.append(config_h)
            start_obj_list.append(config_o)
            print(config_o)
        ibsp = env.ibs[:,[0,1,2,11]] # [x, y, z, components]
        o_list.append(config_o)
        h_list.append(config_h)
        ibs_list.append(ibsp)
        action, stop = rl.select_action(state, evaluate=True)
        state, r, done, info = env.step(action, stop)
        step += 1
        episode_reward += r
        if done[0] or step > all_step:
            if done[0]:
                end_pos_list.append(config_h)
                end_obj_list.append(config_o)
            else:
                end_pos_list.append(start_pos_list[-1])
                end_obj_list.append(start_obj_list[-1])
            config_o = env.opose.copy()
            config_h = env.manipulator.get_euler_state()
            ibsp = env.ibs[:,[0,1,2,11]]
            # Store intermediate states
            o_list.append(config_o)
            h_list.append(config_h)
            ibs_list.append(ibsp)
            env_shape = env.cur_shape
            env_angle = env.cur_init
            # Statics
            shape_list.append(env_shape)
            grasp_list.append(env_angle)
            Q1_list.append(info['Q1'])
            Fnum_list.append(info['Fingers'])
            Coll_list.append(info['Collis'])
            success_list.append(info['SuccessGrasp'])
            fp_list.append(info['FinPen'])
            mean_q = (mean_q + info['Q1'])
            mean_f = (mean_f + info['Fingers'] * info['SuccessGrasp'])
            mean_c = (mean_c + info['Collis'])
            mean_sq += info['SuccessGrasp'] * info['Q1']
            mean_pp += (info['FinPen']>0) * info['FinPen']
            all_p = (all_p + float(info['FinPen']>0)) # All pentration
            all_s = (all_s + float(info['SuccessGrasp'])) # All successfulg grasp
            print("Fingers: %d, Q1, %.1f, collision: %.2f"%(info['Fingers'], info['Q1'], info['Collis']))
            ###
            episode_reward = 0
            step = 0
            cur_counter += 1
            print(cur_counter,"/", test_num)
            ###
            o_list = np.array(o_list)
            h_list = np.array(h_list)
            ibs_list = np.array(ibs_list)
            # save animation states
            env.get_state_data()
            # h_list, o_list, labels = env.get_state_data()
            # print(o_list.shape)
            # print(h_list.shape)
            # print(labels.shape)
            # # Store animation states
            # save_url = os.path.join("trajectories/", TEST_PREFIX, env_shape, env_angle)
            # if not os.path.exists(save_url):
            #     os.makedirs(save_url)
            # np.save(save_url+'/o.npy', o_list)
            # np.save(save_url+'/h.npy', h_list)
            # np.save(save_url+'/L.npy', labels)
            # with open(save_url+'/q1', 'w') as f:
            #     f.write("%.6f\n"%(info['Q1']))
            # with open(save_url+'/FinPen', 'w') as f:
            #     f.write("%.6f\n"%(info['FinPen']))
            o_list = []
            h_list = []
            ibs_list = []
            end = time.time()
            print("Time: ", end-start)
            state = env.reset(mode=mode)
            start = end
    mean_q = (mean_q /test_num)
    mean_c = (mean_c / test_num)
    mean_s = (all_s / test_num)
    mean_f = (mean_f /(all_s+1e-10))
    mean_sq = (mean_sq/(all_s+1e-10))
    mean_pp = (mean_pp/(all_p+1e-10))
    print('MeanQ1:\t',mean_q)
    print('MeanFin:\t',mean_f)
    print('MeanColl:\t',mean_c)
    print('MeanSRate:\t',mean_s)
    print("MeanSQ1:\t", mean_sq)
    print("MeanPP:\t", mean_pp)
    end_all =time.time()
    print("Time Sum:", end_all-start_all)
    if not SIMULATION:
        data = {'Shape':shape_list, "Grasp":grasp_list, "Q1":Q1_list, "Fnum":Fnum_list, "Coll":Coll_list, "FinPen":fp_list, "Start": start_pos_list, "End": end_pos_list, "StartObj": start_obj_list, "EndObj": end_obj_list}
    else:
        data = {'Shape':shape_list, "Grasp":grasp_list, "Q1":Q1_list, "Fnum":Fnum_list, "Coll":Coll_list, "FinPen":fp_list, "Start": start_pos_list, "End": end_pos_list, "StartObj": start_obj_list, "EndObj": end_obj_list, "Success": success_list}
    df = pd.DataFrame(data)
    if not quick_check:
        df.to_csv(TEST_PREFIX+"_test.csv")
    else:
        df.to_csv(TEST_PREFIX+"_eval.csv")
    

def load_model():
    global TEST_PREFIX 
    a_path = None
    c_path = None
    model_dir = "models/" 
    a_name = "sac_actor_%s"%(args.model_name)
    c_name = "sac_critic_%s"%(args.model_name)
    TEST_PREFIX = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) + "_" + a_name
    print("Test Case Name:", TEST_PREFIX)
    a_path = os.path.join(model_dir, a_name)
    c_path = os.path.join(model_dir, c_name)
    print("Actor Network Path:", a_path)
    print("Critic Network Path:", c_path)
    rl.load_model(a_path, c_path)


if __name__ == "__main__":
    if TRAIN_MODEL:
        if CONTINUE:
            load_model()
        train()
    else:
        load_model()
        eval(quick_check=args.quick)


