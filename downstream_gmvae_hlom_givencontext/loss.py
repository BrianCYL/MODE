import torch
import torch.nn.functional as F

def Critic_Loss(agent, state, action, reward, next_state, context, done, gamma):
    with torch.no_grad():
        hat_next_a = agent.get_action(next_state, context, add_noise=True, use_actor_target=True)
        q1_t, q2_t = agent.critic_target(next_state, context, hat_next_a)
        q_next = torch.min(q1_t, q2_t)
        q_targ = reward + (1 - done) * gamma * q_next.detach()
        v_targ = q_targ

    q1_p, q2_p = agent.critic(state, context, action)
    value_p = agent.critic.v(state, context)
    critic_loss = F.mse_loss(q1_p, q_targ) + F.mse_loss(q2_p, q_targ) + F.mse_loss(value_p, v_targ)
    return critic_loss

def CVAE_Loss(agent, state, context, action, temp=1.0):
    # adv = agent.critic.Q1(state, context, action) - agent.critic.v(state, context)  # [B, 1]
    adv = (agent.critic.Q1(state, context, action) - agent.critic.v(state, context)) / temp
    adv = torch.tanh(adv).detach()  # [B, 1]
    input = torch.cat([state, action, context, adv], dim=-1)  # [B, s_dim + a_dim + z_dim + 1]
    latent_input = torch.cat([state, context, adv], dim=-1)  # [B, z_dim + 1]
    z2_mu, z2_logvar, _, recon = agent.cvae(input, latent_input)
    rec_loss = torch.pow(recon - action, 2).mean(dim=1)  # regularization term
    kl_loss = -0.5 * (1 + z2_logvar - z2_mu.pow(2) - z2_logvar.exp()).mean(dim=1)
    l_vae = rec_loss + 1.0 * kl_loss
    cvae_loss = l_vae.mean()
    return cvae_loss

def Actor_Loss(agent, state, action, context, beta=1.0, temp=1.0, device="cuda"):
    adv_samples = torch.ones(state.shape[0], 1).to(device)
    best_z2 = agent.actor(torch.cat([state, context, adv_samples], dim=-1))  # [B, z_dim]
    best_a = agent.cvae.decode(torch.cat([state, context, adv_samples], dim=-1), best_z2)  # [B, a_dim]
    Q_a = agent.critic.Q1(state, context, best_a) # [B, 1]
    Q_data = agent.critic.Q1(state, context, action)  # [B, 1]
    lmbda = beta / Q_data.abs().mean().detach()
    # lmbda = beta / Q_a.abs().mean().detach()

    # adv = agent.critic.Q1(state, context, action) - agent.critic.v(state, context)
    adv = (agent.critic.Q1(state, context, action) - agent.critic.v(state, context)) / temp
    adv = torch.tanh(adv).detach()
    z2 = agent.actor(torch.cat((state, context, adv), dim=1))  # [B, z_dim]
    pi = agent.cvae.decode(torch.cat((state, context, adv), dim=1), z2)
    L_reg = torch.pow(action - pi, 2).mean()  # regularization term
    actor_loss = -lmbda * Q_a.mean() + 1.0 * L_reg
    return actor_loss, L_reg

