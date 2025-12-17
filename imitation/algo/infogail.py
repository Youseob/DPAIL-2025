import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from gym import spaces

from .ppo import PPO
from imitation.network import InfoGAILDiscrim
from imitation.utils import RunningMeanNormalizer

class InfoGAIL(PPO):

    def __init__(self, 
                 buffer_exp, 
                 state_shape, 
                 action_shape, 
                 dim_c=24,
                 gamma=0.995, 
                 rollout_length=50000, 
                 mix_buffer=1,
                 batch_size=1000, 
                 use_obs_norm=False,
                 lr_actor=3e-4, 
                 lr_critic=3e-4, 
                 lr_disc=3e-4,
                 units_actor=(64, 64), 
                 units_critic=(64, 64),
                 units_disc=(100, 100), 
                 epoch_ppo=50, 
                 epoch_disc=10,
                 clip_eps=0.2, 
                 lambd=0.97, 
                 coef=1.0,
                 coef_ent=0.0, 
                 max_grad_norm=10.0, 
                 device='cuda', 
                 seed=0,
                 **kwargs):
        super().__init__(
            (state_shape[0]+dim_c, ), action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm, with_class=True,
        )
        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = InfoGAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            dim_c=dim_c,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh()
        ).to(device)
        
        # Observation normalizer
        self.normalizer = None
        if use_obs_norm:
            self.normalizer = RunningMeanNormalizer(state_shape[0])

        self.dim_c = dim_c
        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.coef = coef
        self.CELoss = torch.nn.CrossEntropyLoss()
        
    def sample_code(self, batch_size=1):
        class_label = np.random.randint(low=0, high=self.dim_c, size=(batch_size, ))
        code = np.eye(self.dim_c)[class_label]
        # c = F.one_hot(class_label, self.dim_c).to(self.device) # bs, dim_c
        return class_label[0], code[0]
    
    @torch.no_grad()        
    def step(self, env, state, t, step):
        #sample code_c
        if t == 0:
            self.class_label, self.code = self.sample_code()
        t += 1
        state_ = np.concatenate([state, self.code])
        action, log_pi = self.explore(state_)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state_, action, reward, mask, log_pi, np.concatenate([next_state, self.code]), self.class_label)
        if done :
            t = 0
            next_state = env.reset()
        
        return next_state, t
    
    def update(self, writer=None):
        self.learning_steps += 1

        for epoch_d in range(self.epoch_disc):
            self.learning_steps_disc += 1
            # Samples from current policy's trajectories
            states, actions, class_label = self.buffer.sample(self.batch_size)[:3]
            states = states[:, :-self.dim_c]
            # Samples from expert's demonstration
            states_exp, actions_exp = self.buffer_exp.sample(self.batch_size)[:2]
            
            if self.normalizer is not None:
                with torch.no_grad():
                    states = self.normalizer.normalize_torch(states, self.device)
                    states_exp = self.normalizer.normalize_torch(states_exp, self.device)
            
            # Update discriminator
            self.update_disc(states,  actions, states_exp, actions_exp, class_label)
            
            # Calculate the running mean and std of a data stream
            if self.normalizer is not None:
                self.normalizer.update(states.cpu().numpy())
                self.normalizer.update(states_exp.cpu().numpy())
        
        # We don't use reward signals here
        states, actions, class_label_gt, _, dones, log_pis, next_states = self.buffer.get()
        
        states_ = states[:, :-self.dim_c]
        if self.normalizer is not None:
            with torch.no_grad():
                states_ = self.normalizer.normalize_torch(states_, self.device)
        
        # Calculate rewards
        scaled_coef = min(1, self.learning_steps * (self.rollout_length / 15000000)) * self.coef
        rewards = self.disc.calculate_reward(states_, actions, class_label_gt, coef=scaled_coef)
        wandb.log({
            'disc/reward_mean': rewards.mean().item(),
            'disc/reward_max': rewards.max().item(), 
            'disc/reward_min': rewards.min().item(), 
            'disc/scaled_coef': scaled_coef
        })

        # Update PPO using estimated rewards
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states)

    def update_disc(self, states, actions, states_exp, actions_exp, class_label):
        # Output of discriminator is (-inf, inf), not [0, 1]
        # logits_pi, pred_c = self.disc(states + 0.1*torch.rand_like(states), 
        #                               actions + 0.1*torch.rand_like(actions))
        # logits_exp, _ = self.disc(states_exp + 0.1*torch.rand_like(states_exp), 
        #                           actions_exp + 0.1*torch.rand_like(actions_exp))
        
        logits_pi, pred_c = self.disc(states, actions)
        logits_exp, _ = self.disc(states_exp, actions_exp)
        
        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)]
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        # Maximize Q(c| s, a)
        loss_c = self.CELoss(pred_c, class_label.view(-1)) # (bs, dim_c)
        loss_disc = loss_pi + loss_exp + loss_c
        
        self.optim_disc.zero_grad()

        loss_disc.backward()
        self.optim_disc.step()
        
        if self.learning_steps_disc % self.epoch_disc == 0:
            # Discriminator's accuracies
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
                acc_c = (torch.argmax(pred_c, dim=-1) == class_label).float().mean()
            
            wandb.log({
                'disc/loss': (loss_pi + loss_exp).item(),
                'disc/acc_pi': acc_pi,
                'disc/acc_exp': acc_exp,
                'disc/c_loss': loss_c.item(),
                'disc/acc_c': acc_c.item()
                })
