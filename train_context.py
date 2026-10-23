import pickle
import numpy as np
import torch
import argparse
import gym
import os
import time
import math
import json

from src.envs import PointEnv, HalfCheetahVelEnv, HalfCheetahDirEnv, AntDirEnv, HopperRandParamsEnv, WalkerRandParamsWrappedEnv, ReachEnv

start_time = time.time()
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

from context.model import RNNContextGMVAE, StateDecoder, RewardDecoder
from context.dataset import ContextDataset
from configs import args_point_robot ,args_half_cheetah_vel, args_half_cheetah_dir, args_ant_dir, args_hopper, args_walker, args_reach

def kl_cat_logits(q_logits, p_logits):
    # KL( softmax(q_logits) || softmax(p_logits) )
    logq = F.log_softmax(q_logits, dim=-1)
    logp = F.log_softmax(p_logits, dim=-1)
    q = logq.exp()
    return (q * (logq - logp)).sum(dim=-1)  # per-sample

def kl_diag_gauss(mu_q, logvar_q, mu_p, logvar_p):
    # All shapes [..., Dz]
    var_q, var_p = logvar_q.exp(), logvar_p.exp()
    return 0.5 * ((logvar_p - logvar_q) + (var_q + (mu_q - mu_p)**2)/var_p - 1).sum(dim=-1)

def gaussian_nll(x, mu, logvar, reduce_dims=(-1,)):
    """
    Negative log-likelihood for a diagonal Gaussian.
    
    x, mu, logvar : [..., D] tensors (same shape)
    reduce_dims   : tuple of axes to sum over (e.g., feature dim)
    Returns:
        per-sample NLL (scalar if reduced)
    """
    # Optional clamp to avoid overflow
    logvar = torch.clamp(logvar, min=-2.0, max=2.0)
    # 0.5 * [(x-mu)^2 / var + log var + log(2π)]
    return 0.5 * ((x - mu)**2 * torch.exp(-logvar) + logvar + math.log(2 * math.pi)).sum(dim=reduce_dims)

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
    env.load_all_tasks(np.load('./datasets/PointRobot-v0/task_goals.npy'))
elif env_type =='cheetah_vel':
   
    with open('./datasets/HalfCheetahVel-v0/task_goals.pkl', 'rb') as file:
        velocities = pickle.load(file)
    # velocities = np.array([item['velocity'] for item in velocities])
    # tasks = [{'velocity': velocity} for velocity in velocities]
    env=HalfCheetahVelEnv(tasks=velocities)
elif env_type=='ant_dir':
    with open('./datasets/AntDir-v0/task_goals.pkl', 'rb') as file:
        velocities = pickle.load(file)
    # velocities = np.array([item['goal'] for item in velocities])


    # tasks = [{'goal': velocity} for velocity in velocities]
    env=AntDirEnv(tasks=velocities)
elif env_type=='walker':
    with open('./datasets/WalkerRandParams-v0/task_goals.pkl', 'rb') as file:
        goals = pickle.load(file)
    
    env = WalkerRandParamsWrappedEnv(tasks=goals)
elif env_type == 'reach':
    with open('./datasets/Reach/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = ReachEnv(tasks=tasks)
elif env_type == 'cheetah_dir':
    with open('./datasets/HalfCheetahDir-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HalfCheetahDirEnv(tasks=tasks)
elif env_type == 'hopper':
    with open('./datasets/HopperRandParams-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = HopperRandParamsEnv(tasks=tasks)    

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
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


train_dataset = ContextDataset(train_data, state_normalize=True, horizon=args.context_horizon, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=args.context_batch_size, shuffle=True)

state_mean, state_std = train_dataset.state_mean, train_dataset.state_std
test_dataset = ContextDataset(test_data, state_normalize=True, mean=state_mean, std=state_std, horizon=args.context_horizon, device=device)
# test_dataset = ContextDataset(test_data, state_normalize=True, horizon=args.context_horizon, device=device)
test_dataloader = DataLoader(test_dataset, batch_size=args.context_batch_size, shuffle=True)

# Tesnorboard
writer = SummaryWriter(f'runs/{args.env_name}/context/{args.data_quality}/horizon{args.context_horizon}')

# The models
context_encoder = RNNContextGMVAE(state_dim, action_dim, args.context_dim, args.context_hidden_dim, args.gmm_k).to(device)
state_decoder = StateDecoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)
reward_decoder = RewardDecoder(state_dim, action_dim, args.context_dim, args.context_hidden_dim).to(device)

optimizer = torch.optim.Adam([*context_encoder.parameters(), *state_decoder.parameters(), *reward_decoder.parameters()], lr=args.context_lr)

save_model_path = f'saves/{args.env_name}/context_gmvae/{args.data_quality}/{args.seed}/horizon{args.context_horizon}'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

global_step = 0
best_loss = float('inf')
kl_max = args.downstream_kl_div
# kl_weight = 1.0
warmup_steps = int(0.2 * args.context_train_epochs * args.context_batch_size)  # 20% warm-up
entropy_coef = 1e-3

for epoch in tqdm(range(args.context_train_epochs)):
    print(f'\n========== Epoch {epoch+1} ==========')

    # Model training
    context_encoder.train(); reward_decoder.train(); state_decoder.train();
    for step, (transition, segment, next_segment) in tqdm(enumerate(train_dataloader)):
    # for step, (transition, segment, _) in enumerate(test_dataloader):
        state, action, reward, next_state, _, _, _, _, _ = transition 
        state_segment, action_segment, reward_segment = segment
        state_segment, action_segment, reward_segment = state_segment.transpose(0,1), action_segment.transpose(0,1), reward_segment.transpose(0,1)

        output = context_encoder(state_segment, action_segment, reward_segment)
        context = output["z"]                          
        qy, mu_k, logvar_k = output["qy"], output["mu_k"], output["logvar_k"]
        py, mu_p, logvar_p, py_logits = output["prior"]

        # reconstruction loss
        # r_mu, r_logvar = reward_decoder(state, action, context)
        # s_mu, s_logvar = state_decoder(state, action, context)
        # reward_err = gaussian_nll(reward, r_mu, r_logvar).mean()     
        # state_err = gaussian_nll(next_state, s_mu, s_logvar).mean()
        # pred_error = reward_err + state_err
        pred_r = reward_decoder(state, action, context)
        pred_s = state_decoder(state, action, reward, next_state, context)
        r_err = (pred_r - reward).pow(2).mean()
        s_err = (pred_s - next_state).pow(2).mean()
        pred_err = r_err + s_err

        # KL of categorical distribution
        cat_kl = kl_cat_logits(output["logits_y"], py_logits).mean()

        # KL of Gaussian mixture
        # B, K, Dz = mu_k.shape
        # mu_p_b   = mu_p.unsqueeze(0).expand(B, K, Dz)
        # logvar_p_b = logvar_p.unsqueeze(0).expand(B, K, Dz)
        kl_z_per_k = kl_diag_gauss(mu_k, logvar_k, mu_p, logvar_p)   # [B,K]
        E_kl_z = (qy * kl_z_per_k).sum(dim=-1).mean()
        kl_loss = cat_kl + E_kl_z
        
        # entropy regularization of p(y) to avoid collapse
        # encourage exploration in the latent space
        entropy = -(qy * (qy + 1e-8).log()).sum(-1).mean()

        # progress = min(global_step / warmup_steps, 1.0)
        # kl_weight = kl_max * progress

        # loss = pred_err + kl_weight * kl_loss - entropy_coef * entropy

        loss = pred_err + kl_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*context_encoder.parameters(), *reward_decoder.parameters(), *state_decoder.parameters()], 1.0)
        optimizer.step()

        global_step += 1
        writer.add_scalar('loss/train', loss.item(), global_step)
        
    # model evaluation
    reward_error, state_error, pred_error, eval_loss = [], [], [], []
    context_encoder.eval(); reward_decoder.eval()
    for transition, segment, next_segment in test_dataloader:
        with torch.no_grad():
            state, action, reward, next_state, _, _, _, _, _ = transition 
            state_segment, action_segment, reward_segment = segment
            state_segment, action_segment, reward_segment = state_segment.transpose(0,1), action_segment.transpose(0,1), reward_segment.transpose(0,1)

            output = context_encoder(state_segment, action_segment, reward_segment)
            context = output["z"]                          
            qy, mu_k, logvar_k = output["qy"], output["mu_k"], output["logvar_k"]
            py, mu_p, logvar_p, py_logits = output["prior"]

            # reconstruction loss
            # r_mu, r_logvar = reward_decoder(state, action, context)
            # s_mu, s_logvar = state_decoder(state, action, context)
            # reward_err = gaussian_nll(reward, r_mu, r_logvar).mean()     
            # state_err = gaussian_nll(next_state, s_mu, s_logvar).mean()
            # pred_error = reward_err + state_err
            pred_r = reward_decoder(state, action, context)
            pred_s = state_decoder(state, action, reward, next_state, context)
            reward_err = (pred_r - reward).pow(2).mean()
            state_err = (pred_s - next_state).pow(2).mean()
            pred_err = reward_err + state_err

            # KL of categorical distribution
            cat_kl = kl_cat_logits(output["logits_y"], py_logits).mean()

            # KL of Gaussian mixture
            # B, K, Dz = mu_k.shape
            # mu_p_b   = mu_p.unsqueeze(0).expand(B, K, Dz)
            # logvar_p_b = logvar_p.unsqueeze(0).expand(B, K, Dz)
            kl_z_per_k = kl_diag_gauss(mu_k, logvar_k, mu_p, logvar_p)   # [B,K]
            E_kl_z = (qy * kl_z_per_k).sum(dim=-1).mean()
            kl_loss = cat_kl + E_kl_z

            # loss = pred_err + kl_weight * kl_loss - entropy_coef * entropy
            loss = pred_err + kl_loss - entropy_coef * entropy
            eval_loss.append(loss.item())
            pred_error.append(pred_err.item())
            reward_error.append(reward_err.item())
            state_error.append(state_err.item())

    loss = np.mean(eval_loss)
    reward_err = np.mean(reward_error)
    state_err = np.mean(state_error)
    pred_error = np.mean(pred_error)

    writer.add_scalar('loss/test', loss, epoch+1)           
    print(f'Model Evaluation, test loss: {loss}, pred error: {pred_error.item()}, kl loss: {kl_loss.item()}, entropy: {entropy.item()}')
    print(f'Model Evaluation, reward err: {reward_err}, state err: {state_err}')

    if loss < best_loss:
        best_loss = loss 
        torch.save(
            {'context_encoder': context_encoder.state_dict(), 
             'reward_decoder': reward_decoder.state_dict(), 
             'state_decoder': state_decoder.state_dict()}, 
            f'{save_model_path}/context_models_best.pt'
        )
        print('Save the best model...')

    # print(f'Predicted rewards: {r_mu.detach().cpu().numpy()[:8].reshape(-1)}')
    # print(f'Real rewards     : {reward.detach().cpu().numpy()[:8].reshape(-1)}')

    if (epoch+1) % args.save_context_model_every == 0:
        torch.save(
            {'context_encoder': context_encoder.state_dict(), 'reward_decoder': reward_decoder.state_dict()}, 
            f'{save_model_path}/context_models_{epoch+1}.pt'
        )

    print(f'\nElapsed time: {round((time.time()-start_time)/60., 2)} minutes')