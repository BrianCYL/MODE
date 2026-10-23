import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from context.model import RNNContextGMVAE, StateDecoder, RewardDecoder
import os

class Policy(nn.Module):
    def __init__(self, state_dim, ac_dim, z1_dim, hidden_dim, deterministic=False):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.deterministic = deterministic
        self.model = nn.Sequential(nn.Linear(state_dim + z1_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU()
                                  )   
        self.mu_head = nn.Linear(hidden_dim, ac_dim)
        self.logvar_head = nn.Linear(hidden_dim, ac_dim)

    def forward(self, s, context):
        feat = self.model(torch.cat([s, context], dim=-1))
        loc = torch.tanh(self.mu_head(feat))
        if self.deterministic:
            return loc
        else:
            scale = torch.exp(self.logvar_head(feat).clamp(-15, 2))
            normal_dist = Normal(loc, scale)
            return normal_dist, loc

class GatingPolicy(nn.Module):
    def __init__(self, state_dim, z1_dim, a_dim, hidden_dim, temperature=1.0):
        super(GatingPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + z1_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, a_dim) # Outputs K logits
        )
        self.tau = temperature

    def forward(self, state, context):
        return self.model(torch.cat([state, context], dim=-1))

    def get_weights(self, state, context):
        # Input: [B, state_dim], [B, z1_dim]
        # Output: [B, K] (softmax weights)
        logits = self.model(torch.cat([state, context], dim=-1))
        # Use softmax to get a valid probability distribution (our weights 'w')
        return F.softmax(logits / self.tau, dim=-1)

class EnsembleCritic(nn.Module):
    def __init__(self, s_dim, z_dim, a_dim, hidden_dim=128, n_critics=5):
        super().__init__()
        self.n_critics = n_critics
        
        # Create a list of N critics
        self.critics = nn.ModuleList()
        for _ in range(n_critics):
            q_net = nn.Sequential(
                nn.Linear(s_dim + z_dim + a_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), # Added a layer for capacity
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.critics.append(q_net)

    def forward(self, s, z, a):
        # Input: [B, Dims...]
        # Output: [B, N_CRITICS]
        inp = torch.cat([s, z, a], 1)
        # inp = torch.cat([s, z, a, cur_step], 1)
        
        # Get Q-values from all critics
        q_preds = [q_net(inp) for q_net in self.critics]
        q_preds = torch.cat(q_preds, dim=1)
        return q_preds

    def Q1(self, s, z, a):
        inp = torch.cat([s, z, a], 1)
        # inp = torch.cat([s, z, a, cur_step], 1)
        return self.critics[0](inp)

class MixtureGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, z1_dim, num_mixtures, hidden_dim):
        super(MixtureGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z1_dim = z1_dim
        self.num_mixtures = num_mixtures

        self.model = nn.Sequential(nn.Linear(state_dim + z1_dim, hidden_dim), 
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, action_dim * num_mixtures * 2 + num_mixtures)
                                   )
    
    # Return the means, log_stds, weights for each Gaussian components.
    def forward(self, state, context):
        x = self.model(torch.cat([state, context], dim=-1))
        an = self.action_dim * self.num_mixtures
        # means = torch.tanh(x[..., : an].view(-1, self.num_mixtures, self.action_dim))
        # means = torch.clamp(means, -0.9999, 0.9999)
        means = x[..., : an].view(-1, self.num_mixtures, self.action_dim)
        # means = torch.tanh(means)
        log_stds = x[..., an: 2*an].view(-1, self.num_mixtures, self.action_dim)
        # weights = F.softmax(x[..., 2*an: ], dim=-1)
        weights = x[..., 2*an:]
        return means, log_stds, weights

    '''
        This function samples actions from all Gaussian components.
        Input: Tensor state
        Output: Samples with shape (batch_size, num_mixtures, ac_dim)
    '''
    def sample_actions_for_components(self, state, context):
        means, log_stds, _ = self.forward(state, context)
        stds = (log_stds.clamp(-15, 2)).exp()
        # stds = torch.exp(log_stds)
        dist = torch.distributions.Normal(means, stds)
        samples = dist.sample()
        return samples

class Agent():
    def __init__(self, env_type, state_dim, action_dim, action_scale, num_mixtures, num_q_samples, device='cuda', args=None, downstream_epoch=0, context_epoch=None):
        self.env_type = env_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.num_mixtures = num_mixtures
        self.num_q_samples = num_q_samples

        self.device = device
        self.args = args
        self.downstream_epoch = downstream_epoch
        self.load_context_model(context_epoch)

        self.gmm_mix = MixtureGaussianPolicy(state_dim=state_dim, action_dim=action_dim, z1_dim=args.context_dim, num_mixtures=num_mixtures, hidden_dim=args.downstream_h_dim).to(device)
        # self.bc_policy = MixtureGaussianPolicy(state_dim=state_dim, action_dim=action_dim, z1_dim=args.context_dim, num_mixtures=1, hidden_dim=args.downstream_h_dim).to(device)
        self.actor = GatingPolicy(state_dim=state_dim, z1_dim=args.context_dim, a_dim=num_mixtures, hidden_dim=args.downstream_h_dim, temperature=args.temperature).to(device)
        # self.actor = GatingPolicy(state_dim=state_dim, z1_dim=args.context_dim, a_dim=num_mixtures+1, hidden_dim=args.downstream_h_dim).to(device)
        # self.actor = Policy(state_dim=state_dim, ac_dim=action_dim, z1_dim=args.context_dim, hidden_dim=args.downstream_h_dim, deterministic=False).to(device)
        self.critic = EnsembleCritic(s_dim=state_dim, z_dim=args.context_dim, a_dim=action_dim, hidden_dim=args.downstream_h_dim, n_critics=args.n_critics).to(device)
        self.critic_target = EnsembleCritic(s_dim=state_dim, z_dim=args.context_dim, a_dim=action_dim, hidden_dim=args.downstream_h_dim, n_critics=args.n_critics).to(device)        
        self.critic_target.load_state_dict(self.critic.state_dict())

    def load_context_model(self, epoch=None):
        if epoch is not None:
            # load_path = f'./saves/{self.args.env_name}/context_gmvae/medium-expert/3407/horizon{self.args.context_horizon}/context_models_{epoch}.pt'
            load_path = f'./saves/{self.args.env_name}/context_gmvae/{self.args.data_quality}/412/horizon{self.args.context_horizon}/context_models_{epoch}.pt'
            # load_path = f'./saves/{self.args.env_name}/context_gmvae/medium-expert/412/horizon{self.args.context_horizon}/context_models_{epoch}.pt'
        else:
            load_path = f'./saves/{self.args.env_name}/context_gmvae/{self.args.data_quality}/412/horizon{self.args.context_horizon}/context_models_best.pt'
            # load_path = f'./saves/{self.args.env_name}/context_gmvae/medium-expert/412/horizon{self.args.context_horizon}/context_models_best.pt'
            # load_path = f'./saves/{self.args.env_name}/context_gmvae_k1/{self.args.data_quality}/3407/horizon{self.args.context_horizon}/context_models_best.pt'
        # load_path = f'./saves/{self.args.env_name}/context/medium-expert/horizon{self.args.context_horizon}/context_models_best.pt'
        # load_path = f'./saves/{self.args.env_name}/context_vae/{self.args.data_quality}/horizon{self.args.context_horizon}/context_models_10.pt'
        self.context_encoder = RNNContextGMVAE(self.state_dim, self.action_dim, self.args.context_dim, self.args.context_hidden_dim, self.args.gmm_k).to(self.device)
        self.state_decoder = StateDecoder(self.state_dim, self.action_dim, self.args.context_dim, self.args.context_hidden_dim).to(self.device)
        self.reward_decoder = RewardDecoder(self.state_dim, self.action_dim, self.args.context_dim, self.args.context_hidden_dim).to(self.device)
        if os.path.exists(load_path):
            self.context_encoder.load_state_dict(torch.load(load_path)['context_encoder'])
            self.state_decoder.load_state_dict(torch.load(load_path)['state_decoder'])
            self.reward_decoder.load_state_dict(torch.load(load_path)['reward_decoder'])
            print('Load context model from {}'.format(load_path))
        else:
            print("No pretrained context model found, using initial model.")

        for name, param in self.context_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.state_decoder.named_parameters():
            param.requires_grad = False
        for name, param in self.reward_decoder.named_parameters():
            param.requires_grad = False
        
       
    def load_downstream_models(self):
        if self.downstream_epoch is not None:
            load_path = f'./saves/{self.args.env_name}/{self.args.agent_save_folder_name}/{self.args.data_quality}/{self.args.seed}/downstream_models_{self.load_epoch}.pt'
        else:
            load_path = f'./saves/{self.args.env_name}/{self.args.agent_save_folder_name}/{self.args.data_quality}/{self.args.seed}/downstream_models_best.pt'

        if not os.path.exists(load_path):
            print("Not pretrained downstream models found, start training from scratch.")
            return 
        checkpoint = torch.load(load_path)
        self.gmm_mix.load_state_dict(checkpoint['gmm_mix'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        print('Load downstream models from {}'.format(load_path))

    def mode(self, mode):
        if mode == "train":
            self.gmm_mix.train()
            self.actor.train()
            self.critic.train()
        elif mode == "eval":
            self.gmm_mix.eval()
            self.actor.eval()
            self.critic.eval()
        else:
            raise ValueError("Mode must be either 'train' or 'eval'")

    def get_context(self, state_seg, action_seg, reward_seg):
        output = self.context_encoder(state_seg, action_seg, reward_seg, sampling=False)
        context = output['z']  # [B, z1_dim]
        return context.detach()
    
    # def get_action(self, state, context):
    #     with torch.no_grad():
    #         _, action_mean = self.actor(state, context)
    #     return torch.clip(action_mean, -self.action_scale, self.action_scale)


    def get_action(self, state, context):
        with torch.no_grad():
            weights = self.actor.get_weights(state, context) # [B, K]

            # if np.random.rand() < 0.01: # Print 1% of the time
            #     print(f"Test Weights: {weights[0].cpu().numpy().round(2)}")
            # 2. Get the parameters for all K modes
            means, _, _ = self.gmm_mix(state, context) # [B, K, A_dim]
            
            # 3. Compute the synthesized action (weighted average of means)
            # weights.unsqueeze(-1) -> [B, K, 1]
            # (weights * means) -> [B, K, A_dim] (element-wise broadcast)
            # .sum(dim=1) -> [B, A_dim]
            action_mean = (weights.unsqueeze(-1) * means).sum(dim=1)
            
            # 4. Clip the final action to be within bounds
        return torch.clip(action_mean, -self.action_scale, self.action_scale)

    # def get_action(self, state, context):
    #     with torch.no_grad():
    #         # 1. Get K+1 mixture weights from the gating policy
    #         all_weights = self.actor(state, context) # [B, K+1]
            
    #         # Split the weights
    #         weights_gmm = all_weights[:, :self.num_mixtures] # [B, K]
    #         weights_bc = all_weights[:, self.num_mixtures:] # [B, 1]

    #         # 2. Get the parameters for all modes
    #         means_gmm, _, _ = self.gmm_mix(state, context) # [B, K, A_dim]
    #         means_bc, _, _ = self.bc_policy(state, context) # [B, 1, A_dim]
            
    #         # 3. Compute the synthesized action (weighted average of means)
    #         action_gmm = (weights_gmm.unsqueeze(-1) * means_gmm).sum(dim=1)
    #         action_bc = (weights_bc.unsqueeze(-1) * means_bc).sum(dim=1)
            
    #         # Total blended action
    #         action_mean = action_gmm + action_bc
            
    #     # 4. Clip the final action to be within bounds
    #     return torch.clip(action_mean, -self.action_scale, self.action_scale)
    
    def get_lcb(self, model, state, context, use_target_critic=False):
        """
        get LCB values for a given GMM-style model
        """
        with torch.no_grad():
            B, K, M, A = state.shape[0], model.num_mixtures, self.num_q_samples, self.action_dim
            
            means, log_stds, _ = model(state, context)
            stds = (log_stds.clamp(-5, 15)).exp()
            # stds = torch.exp(log_stds)
            
            dist = Normal(means, stds)
            actions_from_modes = dist.rsample((M,)).permute(1, 2, 0, 3) # [B, K, M, A_dim]
            
            actions_flat = actions_from_modes.reshape(B * K * M, A)
            # cur_step_flat = cur_step.unsqueeze(1).unsqueeze(1).expand(B, K, M, -1).reshape(B * K * M, 1)
            state_flat = state.unsqueeze(1).unsqueeze(1).expand(B, K, M, -1).reshape(B * K * M, -1)
            context_flat = context.unsqueeze(1).unsqueeze(1).expand(B, K, M, -1).reshape(B * K * M, -1)

            critic_to_use = self.critic_target if use_target_critic else self.critic
            q_all_ensemble = critic_to_use(state_flat, context_flat, actions_flat)
            # q_all_ensemble = critic_to_use(state_flat, context_flat, actions_flat, cur_step_flat)
            
            q_mean = q_all_ensemble.mean(dim=1)
            q_std = q_all_ensemble.std(dim=1)
            # cur_step_squeezed = cur_step_flat.reshape(-1)
            # lcb = q_mean - self.args.lambda_lcb * (1 + 9.0 * cur_step_squeezed) * q_std
            lcb = q_mean - self.args.lambda_lcb * q_std
            
            lcb_by_mode = lcb.reshape(B, K, M)
            expected_lcb_k = lcb_by_mode.mean(dim=2) # [B, K]
            
        return expected_lcb_k

    # def get_lcb(self, model, state, context, use_target_critic=False):
    #     """
    #     get LCB values for a given GMM-style model
    #     """
    #     with torch.no_grad():
    #         B, K, A = state.shape[0], model.num_mixtures, self.action_dim
            
    #         means, _, _ = model(state, context)
                        
    #         actions_flat = means.reshape(B * K, A)
    #         # cur_step_flat = cur_step.unsqueeze(1).unsqueeze(1).expand(B, K, M, -1).reshape(B * K * M, 1)
    #         state_flat = state.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    #         context_flat = context.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

    #         critic_to_use = self.critic_target if use_target_critic else self.critic
    #         q_all_ensemble = critic_to_use(state_flat, context_flat, actions_flat)
    #         # q_all_ensemble = critic_to_use(state_flat, context_flat, actions_flat, cur_step_flat)
            
    #         q_mean = q_all_ensemble.mean(dim=1)
    #         q_std = q_all_ensemble.std(dim=1)
    #         # cur_step_squeezed = cur_step_flat.reshape(-1)
    #         # lcb = q_mean - self.args.lambda_lcb * (1 + 9.0 * cur_step_squeezed) * q_std
    #         lcb = q_mean - self.args.lambda_lcb * q_std
            
    #         expected_lcb_k = lcb.reshape(B, K)
    #         # expected_lcb_k = lcb_by_mode.mean(dim=1) # [B, K]

    #     return expected_lcb_k

    def save_models(self, epoch):
        save_path = f'./saves/{self.args.env_name}/{self.args.agent_save_folder_name}/{self.args.data_quality}/{self.args.seed}/downstream_models_{epoch}.pt'
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save({
            'gmm_mix': self.gmm_mix.state_dict(),
            # 'bc_policy': self.bc_policy.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, save_path)
        print(f'Save downstream models to {save_path}')

    def to(self):
        self.context_encoder.to(self.device)
        self.gmm_mix.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def train_critic(self, state, action, reward, next_state, context, done, gamma):
        with torch.no_grad():
            # target_actions = self.get_action(next_state, context)  
            # weights = self.actor.get_weights(next_state, context) # [B, K]         
            # noise = (torch.randn_like(weights) * self.args.policy_noise).clamp(-self.args.noise_clip, self.args.noise_clip)

            # target_weights = F.softmax(weights + noise, dim=-1) # [B, K]
            # means, _, _ = self.gmm_mix(next_state, context) # [B, K, A_dim]
            # target_actions = (target_weights.unsqueeze(-1) * means).sum(dim=1) # [B, A_dim] 
            # target_actions = torch.clip(target_actions, -self.action_scale, self.action_scale)
            target_action_clean = self.get_action(next_state, context)
            noise = (torch.randn_like(target_action_clean) * self.args.smooth_noise).clamp(-self.args.noise_clip, self.args.noise_clip)
            target_actions = torch.clamp(target_action_clean + noise, -self.action_scale, self.action_scale)
            next_q_all = self.critic_target(next_state, context, target_actions) # [B, N_CRITICS]
            next_q, _ = torch.min(next_q_all, dim=1, keepdim=True) # [B, 1]
            target_q_value = reward + (1 - done) * gamma * next_q

        q_preds = self.critic(state, context, action) # [B, N_CRITICS]
        critic_loss = F.mse_loss(q_preds, target_q_value.expand_as(q_preds))
        return critic_loss

    # def train_critic(self, state, action, reward, next_state, context, done, gamma, global_step):
    #     # is_actor_training = global_step >= (self.args.warm_train + self.args.gmm_train_steps)
    #     # next_step = cur_step - 1 / self.args.max_episode_steps
    #     with torch.no_grad():
    #         logits = self.actor(next_state, context) # [B, K]         
    #         noise = (torch.randn_like(logits) * self.args.policy_noise).clamp(-self.args.noise_clip, self.args.noise_clip)

    #         target_weights = F.softmax((logits + noise) / self.args.temperature, dim=-1) # [B, K]
    #         means, _, _ = self.gmm_mix(next_state, context) # [B, K, A_dim]
    #         target_actions = (target_weights.unsqueeze(-1) * means).sum(dim=1) # [B, A_dim] 
    #         target_actions = torch.clip(target_actions, -self.action_scale, self.action_scale)
    #         next_q_all = self.critic_target(next_state, context, target_actions) # [B, N_CRITICS]
    #         next_q, _ = torch.min(next_q_all, dim=1, keepdim=True) # [B, 1]
    #         target_q_value = reward + (1 - done) * gamma * next_q

    #     q_preds = self.critic(state, context, action) # [B, N_CRITICS]
    #     critic_loss = F.mse_loss(q_preds, target_q_value.expand_as(q_preds))
    #     return critic_loss

    # def train_critic(self, state, action, reward, next_state, next_action, context, done, gamma):
    #     with torch.no_grad():
    #         # next_logits = self.actor(next_state, context)
    #         # best_mode_idx = torch.argmax(next_logits, dim=1) # [B]

    #         # means_gmm, _, _ = self.gmm_mix(next_state, context) # [B, K, A]
            
    #         # best_mode_idx_expanded = best_mode_idx.view(-1, 1, 1).expand(-1, 1, self.action_dim)
    #         # target_action_clean = torch.gather(means_gmm, 1, best_mode_idx_expanded).squeeze(1) # [B, A]

    #         noise = (torch.randn_like(next_action) * self.args.smooth_noise).clamp(-self.args.noise_clip, self.args.noise_clip)
    #         target_action = torch.clamp(next_action + noise, -self.action_scale, self.action_scale)

    #         # --- 4. Compute Target Q (Standard) ---
    #         next_q_all = self.critic_target(next_state, context, target_action)
    #         next_q, _ = torch.min(next_q_all, dim=1, keepdim=True)
    #         target_q_value = reward + (1 - done) * gamma * next_q

    #     # --- 5. Compute Loss ---
    #     q_preds = self.critic(state, context, action)
    #     critic_loss = F.mse_loss(q_preds, target_q_value.expand_as(q_preds))
        
    #     return critic_loss

    # def train_critic(self, state, action, reward, next_state, context, done, gamma, global_step):
    #     with torch.no_grad():
    #         # --- HARD SELECTION TARGET ---
            
    #         # 1. Ask the Actor: "Which mode would you pick?"
    #         next_probs = self.actor(next_state, context)
    #         best_mode_idx = torch.argmax(next_probs, dim=1) # [B]

    #         # 2. Get parameters for ALL modes (GMM + BC if used)
    #         means_gmm, _, _ = self.gmm_mix(next_state, context)

    #         # 3. Gather the specific mean for the selected mode
    #         # We select the (B, 1, A) block corresponding to the indices
    #         best_mode_idx_expanded = best_mode_idx.view(-1, 1, 1).expand(-1, 1, self.action_dim)
    #         target_action_mean = torch.gather(means_gmm, 1, best_mode_idx_expanded).squeeze(1)
            
    #         # 4. Add Noise (Standard TD3 smoothing)
    #         noise = (torch.randn_like(target_action_mean) * self.args.policy_noise).clamp(-self.args.noise_clip, self.args.noise_clip)
    #         target_actions = torch.clamp(target_action_mean + noise, -self.action_scale, self.action_scale)
            
    #         # --- END HARD SELECTION ---

    #         # 5. Standard Critic Update
    #         next_q_all = self.critic_target(next_state, context, target_actions)
    #         next_q, _ = torch.min(next_q_all, dim=1, keepdim=True)
    #         target_q_value = reward + (1 - done) * gamma * next_q

    #     q_preds = self.critic(state, context, action)
    #     critic_loss = F.mse_loss(q_preds, target_q_value.expand_as(q_preds))
    #     return critic_loss

    def train_actor(self, state, context):
        """
        This is the new policy training logic.
        It trains the GatingPolicy to maximize the LCB of the synthesized mixture.
        """
        
        # Get the K+1 weights [B, K] from our policy
        # This is the *only* part of this calculation that tracks gradients
        weights = self.actor.get_weights(state, context)
        # print("weights", weights.shape)
        # Get the expected LCB for each of the K modes and the BC policy
        # This part is done with no gradient tracking
        with torch.no_grad():
            expected_lcb_k = self.get_lcb(self.gmm_mix, state, context) # [B, K]
            # means, _, _ = self.gmm_mix(state, context) # [B, K, A_dim]
        # Compute the final policy objective
        # This is the expected LCB of the *full mixture* policy
        # (weights * expected_lcb_k) -> [B, K]
        # .sum(dim=1) -> [B]
        expected_lcb_policy = (weights * expected_lcb_k).sum(dim=1)
        # action_pi = torch.sum(weights.unsqueeze(-1) * means, dim=1)
        # action_pi = torch.clip(action_pi, -self.action_scale, self.action_scale)
        # behavior cloning loss
        # bc_loss = F.mse_loss(action_pi, action)
        # combine lcb loss and bc loss
        
        # The loss is the negative of this objective (we want to maximize it)
        policy_loss = -expected_lcb_policy.mean()
        # lmbda = 1.0 / expected_lcb_policy.abs().mean().detach()
        # policy_loss = -lmbda * expected_lcb_policy.mean() + 10.0 * bc_loss
        # info = {
        #     "expected_lcb_policy": expected_lcb_policy.mean().item(),
        #     "bc_loss": bc_loss.item(),
        # }

        return policy_loss

    # def train_actor(self, state, context):
    #     logits = self.actor(state, context) 
    #     with torch.no_grad():
    #         all_lcb = self.get_lcb(self.gmm_mix, state, context)
    #         target_labels = torch.argmax(all_lcb, dim=1) # [B]
    #     # if np.random.rand() < 0.01: # Print 1% of the time
    #     #     print(f"LCBs: {all_lcb[0].cpu().numpy()}")
            
    #     actor_loss = nn.CrossEntropyLoss()(logits, target_labels)
    #     return actor_loss

    # def train_actor(self, state, context):
    #     _, gen_action = self.actor(state, context) 
    #     with torch.no_grad():
    #         all_lcb = self.get_lcb(self.gmm_mix, state, context)
    #         target_labels = torch.argmax(all_lcb, dim=1) # [B]
    #         chosen_means, _, _ = self.gmm_mix(state, context)
    #         chosen_means_expanded = target_labels.view(-1, 1, 1).expand(-1, 1, self.action_dim)
    #         # print("Chosen means shape:", chosen_means_expanded.shape)
    #         # print("gen action shape:", gen_action.shape)
    #         noise = (torch.randn_like(gen_action) * self.args.policy_noise).clamp(-self.args.noise_clip, self.args.noise_clip)
    #         target_actions_clean = torch.gather(chosen_means, 1, chosen_means_expanded).squeeze(1) # [B, A]
    #         target_actions = torch.clip(target_actions_clean + noise, -self.action_scale, self.action_scale)
    #         q_val = self.critic.Q1(state, context, gen_action)
    #         cur_q = self.critic.Q1(state, context, target_actions)
    #         adv = (q_val - cur_q).detach()
    #         weights = torch.clip(torch.exp(adv * self.args.beta), 0, self.args.C)
    #     # print("weights", weights.shape)
    #     # print("gen_action", gen_action.shape)
    #     # print("target_actions", target_actions.shape)
    #     # print("chosen_means", chosen_means_expanded.shape)

    #     actor_loss = ((gen_action - target_actions).pow(2).mean(dim=1) * weights).mean()
    #     return actor_loss

    # def train_actor(self, state, context):
    #     probs = self.actor(state, context)
    #     log_probs = torch.log(probs + 1e-8) # [B, K] or [B, K+1]
        
    #     # 2. Get the TARGET labels (The mode with the best LCB)
    #     with torch.no_grad():
    #         # Calculate LCB for GMM modes
    #         gmm_lcb = self.get_lcb(self.gmm_mix, state, context) # [B, K]
                        
    #         # Find the index of the best mode for each state
    #         # labels shape: [B]
    #         best_mode_labels = torch.argmax(gmm_lcb, dim=1)

    #     # 3. Compute Classification Loss (NLLLoss takes log_probs and labels)
    #     actor_loss = nn.NLLLoss()(log_probs, best_mode_labels)
        
    #     return actor_loss

    # def train_actor(self, state, context):
    #     """
    #     This is the new policy training logic.
    #     It trains the GatingPolicy to maximize the LCB of the synthesized mixture.
    #     """
        
    #     # Get the K+1 weights [B, K+1] from our policy
    #     # This is the *only* part of this calculation that tracks gradients
    #     weights = self.actor(state, context)
        
    #     # Get the expected LCB for each of the K modes and the BC policy
    #     # This part is done with no gradient tracking
    #     with torch.no_grad():
    #         gmm_lcb = self.get_lcb(self.gmm_mix, state, context) # [B, K]
    #         bc_lcb = self.get_lcb(self.bc_policy, state, context) # [B, 1]
    #         expected_lcb_k = torch.cat([gmm_lcb, bc_lcb], dim=1) # [B, K+1]

    #     # Compute the final policy objective
    #     # This is the expected LCB of the *full mixture* policy
    #     # (weights * expected_lcb_k) -> [B, K+1]
    #     # .sum(dim=1) -> [B]
    #     expected_lcb_policy = (weights * expected_lcb_k).sum(dim=1)
        
    #     # The loss is the negative of this objective (we want to maximize it)
    #     policy_loss = -expected_lcb_policy.mean()
        
    #     return policy_loss

    def train_gmm_mix(self, state, context, action):
        means, log_stds, weights = self.gmm_mix(state, context)
        stds = (log_stds.clamp(-15, 2)).exp()
        # stds = torch.exp(log_stds)
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(action.unsqueeze(1).expand_as(means))  # [B, num_mixtures, a_dim]
        log_probs = log_probs.sum(dim=-1)                    # [B,M]

        # 3) stable mixture log-weights
        log_w = F.log_softmax(weights, dim=-1)         # [B,M]

        # 4) mixture log-likelihood
        log_likelihood = torch.logsumexp(log_w + log_probs, dim=-1)  # [B]
        gmm_loss = - log_likelihood.mean()

        # with torch.no_grad():
        #     logs = {
        #         "log_likelihood": log_likelihood.mean().item(),   # raw LL (can be negative or positive)
        #         "gmm_loss": gmm_loss.item(),                    # what you should minimize
        #         "logstd": log_stds.mean().item(),
        #     }
        return gmm_loss
    
    # def train_bc_policy(self, state, context, action):
    #     means, log_stds, weights = self.bc_policy(state, context) # means is [B, 1, A_dim]
    #     stds = (log_stds.clamp(-5, 15)).exp()
    #     # stds = torch.exp(log_stds)
    #     dist = torch.distributions.Normal(means, stds)

    #     log_probs = dist.log_prob(action.unsqueeze(1).expand_as(means))
    #     log_probs = log_probs.sum(dim=-1) # [B, 1]

    #     log_w = F.log_softmax(weights, dim=-1) # [B, 1]
    #     log_likelihood = torch.logsumexp(log_w + log_probs, dim=-1)  # [B]
    #     bc_loss = - log_likelihood.mean()
        
    #     return bc_loss