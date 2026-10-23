import numpy as np
import torch
from collections import OrderedDict

import torch.testing


def evaluate(
        env,
        state_dim,
        action_dim,
        agent,
        prompt,
        mean,
        std,
        max_episode_steps=1000,
        device='cuda',
        args = None,
        ):

    agent.mode("eval")

    avg_epi_return = 0.
    avg_epi_len = 0
    
    for _ in range(args.num_eval_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        cur_state = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        context = prompt.to(device=device, dtype=torch.float32)
        # context = torch.zeros((1, args.context_dim), device=device, dtype=torch.float32)
        actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        
        states_traj = np.zeros((args.max_episode_steps, env.observation_space.shape[0]))
        actions_traj = np.zeros((args.max_episode_steps, env.action_space.shape[0]))
        rewards_traj = np.zeros((args.max_episode_steps, 1))

        for t in range(max_episode_steps):
            if args.state_normalize:
                action = agent.get_action((cur_state - mean) / std, context)
            else:
                action = agent.get_action(cur_state, context)
            # print("current action:", action)
            actions = torch.cat([actions, action], dim=0)
            action = action.detach().cpu().numpy()

            step_result = env.step(action)
            state = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            
            states_traj[t] = np.copy(states[-1].detach().cpu().numpy().reshape(-1))
            actions_traj[t] = np.copy(action)
            rewards_traj[t] = np.copy(reward)

            cur_state = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            states = torch.cat([states, cur_state], dim=0)

            state_seg = states_traj[t+1-args.context_horizon : t+1]
            action_seg = actions_traj[t+1-args.context_horizon : t+1]
            reward_seg = rewards_traj[t+1-args.context_horizon : t+1]

            length_gap = args.context_horizon - state_seg.shape[0] 
            state_seg = np.pad(state_seg, ((length_gap, 0),(0, 0)))
            action_seg = np.pad(action_seg, ((length_gap, 0),(0, 0)))
            reward_seg = np.pad(reward_seg, ((length_gap, 0),(0, 0)))

            state_seg = torch.FloatTensor(state_seg).to(device).unsqueeze(1)
            action_seg = torch.FloatTensor(action_seg).to(device).unsqueeze(1)
            reward_seg = torch.FloatTensor(reward_seg).to(device).unsqueeze(1)

            if t >= args.context_horizon:
                context = agent.get_context(state_seg, action_seg, reward_seg) # only update context when the rolling window is full
            
            if args.env_name=='Reach':
                reward = torch.from_numpy(reward).type(torch.cuda.FloatTensor)               

            reward = torch.tensor(reward).to(device=device, dtype=torch.float32)
            rewards = torch.cat([rewards, reward.unsqueeze(0)], dim=0)

            avg_epi_return += reward
            avg_epi_len += 1

            if done:
                break
        

        # trajectory = torch.cat((states[:-1], actions, rewards.reshape(-1,1)), dim=-1).detach().cpu().numpy()
    trajectory = OrderedDict([
    ('observations', states[:-1].cpu().detach().numpy()),
    ('actions', actions.cpu().detach().numpy()),
    ('rewards', rewards.cpu().detach().numpy()),
    ('next_observations', states[1:].cpu().detach().numpy()),
])

    return avg_epi_return/args.num_eval_episodes, avg_epi_len/args.num_eval_episodes, trajectory
