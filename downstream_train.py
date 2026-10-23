from src.envs import PointEnv, HalfCheetahVelEnv, HalfCheetahDirEnv, AntDirEnv, HopperRandParamsEnv, WalkerRandParamsWrappedEnv, ReachEnv
import pickle
import numpy as np
import torch
import argparse
import gym
import os
import time
start_time = time.time()
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from context.dataset import ContextDataset, sample_context_by_task
from configs import args_point_robot ,args_half_cheetah_vel, args_half_cheetah_dir, args_ant_dir, args_hopper, args_walker, args_reach
from downstream_gmvae_hlom_givencontext.agent import Agent
from downstream_gmvae_hlom_givencontext.evaluation import evaluate
from utils import soft_update
import json

parser = argparse.ArgumentParser()
parser.add_argument('--env_type', default='ant_dir')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--context_horizon', type=int, default=4)
args, rest_args = parser.parse_known_args()
env_type = args.env_type
context_horizon = args.context_horizon
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

if env_type == 'point_robot':
    args = args_point_robot.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'cheetah_vel':
    args = args_half_cheetah_vel.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'cheetah_dir':
    args = args_half_cheetah_dir.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'ant_dir':
    args = args_ant_dir.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'hopper':
    args = args_hopper.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'walker':
    args = args_walker.get_args(rest_args)
    args.context_horizon = context_horizon
elif  env_type == 'reach':
    args = args_reach.get_args(rest_args)
    args.context_horizon = context_horizon
else:
    raise NotImplementedError

torch.manual_seed(args.seed)
np.random.seed(args.seed)
np.set_printoptions(precision=4, suppress=True)

# Environment
# make env, multi-task setting
if env_type == 'point_robot':
    env = PointEnv(max_episode_steps=args.max_episode_steps, num_tasks=args.num_tasks)
    env.seed(args.seed)
    env.load_all_tasks(np.load(f'./datasets/PointRobot-v0/task_goals.npy'))
elif env_type =='cheetah_vel':
   
    with open(f"./datasets/HalfCheetahVel-v0/task_goals.pkl", 'rb') as file:
        velocities = pickle.load(file)
    # velocities = np.array([item['velocity'] for item in velocities])
    # tasks = [{'velocity': velocity} for velocity in velocities]
    env=HalfCheetahVelEnv(tasks=velocities)
elif env_type=='ant_dir':
    with open(f"./datasets/AntDir-v0/task_goals.pkl", 'rb') as file:
        velocities = pickle.load(file)
    # velocities = np.array([item['goal'] for item in velocities])


    # tasks = [{'goal': velocity} for velocity in velocities]
    env=AntDirEnv(tasks=velocities)
elif env_type=='walker':
    with open(f'./datasets/WalkerRandParams-v0/task_goals.pkl', 'rb') as file:
        goals = pickle.load(file)
    
    env = WalkerRandParamsWrappedEnv(tasks=goals)
elif env_type == 'reach':
    with open(f"./datasets/Reach/task_goals.pkl", 'rb') as fp:
        tasks = pickle.load(fp)
    env = ReachEnv(tasks=tasks)
elif env_type == 'cheetah_dir':
    with open(f"./datasets/HalfCheetahDir-v0/task_goals.pkl", 'rb') as fp:
        tasks = pickle.load(fp)
    env = HalfCheetahDirEnv(tasks=tasks)
elif env_type == 'hopper':
    with open(f"./datasets/HopperRandParams-v0/task_goals.pkl", 'rb') as fp:
        tasks = pickle.load(fp)
    env = HopperRandParamsEnv(tasks=tasks)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = args.action_scale  # assuming action space is symmetric
num_mixtures = args.num_mixtures
num_q_samples = args.num_q_samples
env.seed(args.seed)
if ((env_type!='walker')and(env_type!='hopper')):
    env.action_space.seed(args.seed)


########################################################################
### preparing the training and testing datasets
data_path = f'datasets/{args.env_name}/{args.data_quality}'
with open(f'datasets/{args.env_name}/{args.data_quality}/task_info.json', 'r') as f:
    task_info = json.load(f)
f.close()
test_task_ids = np.arange(args.num_train_tasks, args.num_tasks)
train_data, test_data = OrderedDict(), OrderedDict()

keys = ['observations', 'actions', 'rewards', 'next_observations', 'next_actions', 'terminals', 'masks', 'task_ids', 'cur_step']
if env_type=='cheetah_dir':
    if args.data_quality=='medium':
        keys = ['states', 'actions', 'rewards', 'next_states', 'next_actions', 'dones', 'masks', 'cur_step']
for key in keys:
    train_data[key] = []
    test_data[key] = []
for task_id in range(args.num_tasks):
    with open(f'{data_path}/dataset_task_{task_id}.pkl', "rb") as f:
        data = pickle.load(f)
    f.close()

    terminal_key = 'dones' if (env_type=='cheetah_dir' and args.data_quality=='medium') else 'terminals'
    
    terminals = np.squeeze(data[terminal_key])
    num_transitions = len(terminals)

    # create next actions
    actions = data['actions']
    next_actions = np.zeros_like(actions)
    next_actions[:-1] = actions[1:]
    next_actions[terminals.astype(bool)] = 0
    indices = np.arange(num_transitions)
    timeout_mask = (indices + 1) % args.max_episode_steps == 0
    next_actions[timeout_mask] = 0
    data['next_actions'] = next_actions

    # add current step index feature
    raw_steps = np.zeros(num_transitions, dtype=np.int32)
    step_counter = 0
    for i in range(num_transitions):
        raw_steps[i] = step_counter / args.max_episode_steps  # Normalize to [0, 1] range
        if i % args.max_episode_steps == args.max_episode_steps - 1 or terminals[i]:
            step_counter = 0 
        else:
            step_counter = min(step_counter + 1, args.max_episode_steps-1) 
    
    data['cur_step'] = raw_steps[:, None]


    for key, values in data.items():
        if task_id < args.num_train_tasks:
            train_data[key].append(values)
        else:
            test_data[key].append(values)
    if task_id < args.num_train_tasks:
        train_data['task_ids'].append(np.array([[task_id for _ in range(len(data['observations']))]]))
    else:
        test_data['task_ids'].append(np.array([[task_id for _ in range(len(data['observations']))]]))

for key, values in train_data.items():
    train_data[key] = np.concatenate(train_data[key], axis=0)
    test_data[key] = np.concatenate(test_data[key], axis=0)
if env_type=='cheetah_dir':
    if args.data_quality=='medium':
        train_data['observations']=train_data['states']
        train_data['next_observations']=train_data['next_states']
        train_data['terminals']=train_data['dones']
        test_data['observations'] = test_data['states']
        test_data['next_observations'] = test_data['next_states']
        test_data['terminals'] = test_data['dones']

########################################################################


train_dataset = ContextDataset(train_data, args.state_normalize, horizon=args.context_horizon, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=args.downstream_batch_size, shuffle=True)

state_mean, state_std = train_dataset.state_mean, train_dataset.state_std

test_dataset = ContextDataset(test_data, args.state_normalize, state_mean, state_std, horizon=args.context_horizon, device=device)
# test_dataset = ContextDataset(test_data, args.state_normalize, horizon=args.context_horizon, device=device)
test_dataloader = DataLoader(test_dataset, batch_size=args.downstream_batch_size, shuffle=False)

# Tensorboard
writer = SummaryWriter(f'runs/{args.env_name}/{args.agent_save_folder_name}_given_context/{args.data_quality}/{args.seed}')

# The models
agent = Agent(env_type, state_dim, action_dim, action_scale=action_scale, num_mixtures=num_mixtures, num_q_samples=num_q_samples, device=device, args=args)

gmm_mix_optimizer = torch.optim.Adam(agent.gmm_mix.parameters(), lr=args.downstream_lr)
# bc_optimizer = torch.optim.Adam(agent.bc_policy.parameters(), lr=1e-5)
actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.downstream_lr)
critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=args.downstream_lr)

actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=10000, gamma=0.9)
critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=10000, gamma=0.9)
gmm_mix_scheduler = torch.optim.lr_scheduler.StepLR(gmm_mix_optimizer, step_size=10000, gamma=0.9)
# bc_scheduler = torch.optim.lr_scheduler.StepLR(bc_optimizer, step_size=10000, gamma=0.9)

gamma = args.gamma

global_step = args.start_step
# global_step = 150000
best_loss = float('inf')
best_gmm_loss = float('inf')
# best_gmm_bc_loss = float('inf')
best_epi_return = float('-inf')
while global_step < args.max_step:
    print(f'\n========== Step {global_step} ==========')

    # Model training
    agent.mode('train')
    for step, (transition, segment, next_segment) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    # for step, (transition, segment, next_segment) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        state, action, reward, next_state, next_action, _, done, _, cur_step = transition 
        state_segment, action_segment, reward_segment = segment
    
        context = agent.get_context(state_segment.transpose(0,1), action_segment.transpose(0,1), reward_segment.transpose(0,1))

        #### GMM loss ####
        train_gmm_mix = global_step < args.gmm_train_steps
        if train_gmm_mix:
            gmm_mix_loss = agent.train_gmm_mix(state, context, action)
            gmm_mix_optimizer.zero_grad()
            gmm_mix_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.gmm_mix.parameters(), 10.0)
            gmm_mix_optimizer.step()
            # if args.scheduler:
            gmm_mix_scheduler.step()

        elif global_step == args.gmm_train_steps: # load the best gmm model
            # load_path = f'saves/{args.env_name}/{args.agent_save_folder_name}/{args.data_quality}/{args.seed}/downstream_models_150000.pt'
            load_path = f'./saves/{args.env_name}/agent_gmvae_hlom_revised/{args.data_quality}/{args.seed}/downstream_models_150000.pt'
            # load_path = f'./saves/{args.env_name}/{args.agent_save_folder_name}/{args.data_quality}/3407/downstream_models_150000.pt'
            print(f'Loading GMM model from {load_path} ...')
            agent.gmm_mix.load_state_dict(torch.load(load_path)['gmm_mix'])
            # agent.bc_policy.load_state_dict(torch.load(load_path)['bc_policy'])
        else:
            pass
        
        #### critic loss ####
        train_critic = global_step >= args.gmm_train_steps
        if train_critic:
            critic_loss = agent.train_critic(state, action, reward, next_state, context, done, gamma)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 10.0)
            critic_optimizer.step()
            if args.scheduler:
                critic_scheduler.step()
        
        #### actor loss ####
        train_actor = global_step >= (args.warm_train + args.gmm_train_steps)
        if train_actor:
            actor_loss = agent.train_actor(state, context)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
            actor_optimizer.step()
            if args.scheduler:
                actor_scheduler.step()
        
        if global_step % args.target_update_freq == 0:
            soft_update(agent.critic_target, agent.critic, tau=args.tau)

        # model evaluation loss
        if (global_step+1) % args.eval_step == 0 and not train_gmm_mix:
            agent.mode('eval')
            with torch.no_grad():
                agent.mode('eval')
                transition, segment, next_segment = next(iter(test_dataloader))
                state, action, reward, next_state, next_action, _, done, _, cur_step = transition 
                state_segment, action_segment, reward_segment = segment

                context = agent.get_context(state_segment.transpose(0, 1), action_segment.transpose(0, 1), reward_segment.transpose(0, 1))
                
                critic_loss = agent.train_critic(state, action, reward, next_state, context, done, gamma)
                actor_loss = agent.train_actor(state, context)
                loss = critic_loss + actor_loss

                writer.add_scalar('critic_loss/test', critic_loss.item(), global_step+1)
                writer.add_scalar('actor_loss/test', actor_loss.item(), global_step+1)          
                print(f'Model Evaluation, test total loss: {loss:.4f}, critic loss: {critic_loss:.4f}, actor loss: {actor_loss:.4f}')

                print(f'\n========== Evaluation at Step {global_step} ==========')
                


                agent.mode('eval')
                avg_epi_return = 0.0
                avg_epi_length = 0.0
                for task_id in test_task_ids:
                    c_states, c_actions, c_rewards = sample_context_by_task(
                                                test_dataloader.dataset, 
                                                task_id, 
                                                context_size=1
                                            )                    
                    with torch.no_grad():
                        context = agent.get_context(c_states.transpose(0,1), c_actions.transpose(0,1), c_rewards.transpose(0,1))

                    env.reset_task(task_id)
                    epi_return, epi_length, eval_traj = evaluate(env, state_dim, action_dim, agent, context, state_mean, state_std, 
                                                    args.max_episode_steps, device,  
                                                    args)
                    avg_epi_return += epi_return
                    avg_epi_length += epi_length
                    print(f'Evaluate on the {task_id}-th test task, received return {epi_return:.2f}')

                avg_epi_return /= len(test_task_ids)
                avg_epi_length /= len(test_task_ids)
                writer.add_scalar('return/test', avg_epi_return, global_step+1)
                writer.add_scalar('length/test', avg_epi_length, global_step+1)
                print(f'Average Return: {avg_epi_return}, Average Length: {avg_epi_length}')

                # agent.save_models(global_step+1)
                if best_epi_return < avg_epi_return:
                    best_step = global_step + 1
                    best_epi_return = avg_epi_return
                    best_length = avg_epi_length
                    print(f'New best model at step {best_step}, return {best_epi_return}')
                    # agent.save_models('best')

        # elif (global_step+1) % args.eval_step == 0 and train_gmm_mix:
        if (global_step+1) % args.eval_step == 0 and train_gmm_mix:
            with torch.no_grad():
                agent.mode('eval')
                gmm_eval_loss = 0.0
                # bc_eval_loss = 0.0
                for transition, segment, next_segment in tqdm(test_dataloader, total=len(test_dataloader)):
                    state, action, reward, next_state, next_action, _, done, _, cur_step = transition 
                    state_segment, action_segment, reward_segment = segment
                    context = agent.get_context(state_segment.transpose(0, 1), action_segment.transpose(0, 1), reward_segment.transpose(0, 1))
                    gmm_mix_loss = agent.train_gmm_mix(state, context, action)
                    # bc_loss = agent.train_bc_policy(state, context, action)
                    gmm_eval_loss += gmm_mix_loss.item()
                    # bc_eval_loss += bc_loss.item()

            gmm_mix_loss = gmm_eval_loss / len(test_dataloader)
            # bc_loss = bc_eval_loss / len(test_dataloader)
            # gmm_bc_loss = gmm_mix_loss + bc_loss
            writer.add_scalar('gmm_mix_loss/test', gmm_mix_loss, global_step+1)
            # writer.add_scalar('bc_loss/test', bc_loss, global_step+1)
            # print(f'GMM Mix Evaluation, test gmm mix loss: {gmm_mix_loss:.4f}, bc loss: {bc_loss:.4f}')
            print(f'GMM Mix Evaluation, test gmm mix loss: {gmm_mix_loss:.4f}')
            agent.save_models(global_step+1)
            if best_gmm_loss > gmm_mix_loss and global_step >= 1e4:
                best_step = global_step + 1
                best_gmm_loss = gmm_mix_loss
                print(f'New best model at step {best_step}, gmm mix loss {gmm_mix_loss:.4f}')
                agent.save_models('best')

        global_step += 1

    if not train_gmm_mix and train_critic and train_actor:
        loss = critic_loss + actor_loss
        writer.add_scalar('critic_loss/train', critic_loss.item(), global_step)
        writer.add_scalar('actor_loss/train', actor_loss.item(), global_step)
    elif train_gmm_mix:
        loss = gmm_mix_loss
        writer.add_scalar('gmm_mix_loss/train', gmm_mix_loss.item(), global_step)
    
    print(f'\nElapsed time: {round((time.time()-start_time)/60., 2)} minutes')