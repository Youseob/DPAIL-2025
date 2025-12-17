import wandb
import torch, copy, os
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from .base import Algorithm
from imitation.buffer import RolloutTrajBuffer
from imitation.network import StateIndependentPolicy, GaussianPolicy
from imitation.utils.utils import EMA

class BC(Algorithm):

    def __init__(self,
                 buffer_exp,
                 state_shape, 
                 action_shape,
                 rollout_length=10000, 
                 batch_size=1000,
                 horizon=4,
                 lr_actor=3e-4, 
                 units_actor=(64, 64),
                 n_pi_epochs=100, 
                 coef_ent=0.0,
                 max_grad_norm=10.0, 
                 ema_decay=0.1,
                 device='cuda',
                 seed=0,
                 **kwargs):
        super().__init__(state_shape, action_shape, device, seed, gamma=None)
        
        # Expert's buffer
        self.buffer_exp = buffer_exp

        # Rollout buffer
        self.buffer = RolloutTrajBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )

        # Actor
        self.actor = GaussianPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU()
        ).to(device)
        
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.horizon = horizon
        self.n_pi_epochs = n_pi_epochs
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.learning_steps = 0
        self.learning_steps_bc = 0

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1
        action, _ = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        self.buffer.append(state, action, reward, done)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        # (bs, horizon, dim)
        for _ in range(self.n_pi_epochs):
            exp_states, exp_actions = self.buffer_exp.sample_traj(batch_size=self.batch_size, horizon=self.horizon)
            self.update_actor(exp_states, exp_actions)
    
        self.buffer.clear()
        
    def update_actor(self, states, actions, eps=1e-8):
        # (bs, horizon, 1)
        self.learning_steps_bc += 1
        target_log_exps = self.actor.evaluate_log_pi(states, actions) 
        # E_exp
        loss_exp = -target_log_exps.sum(1)
        loss_actor = loss_exp.mean()
        # gradient on param on exp_actor
        self.optim_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_bc % self.n_pi_epochs == 0:
            print(f'{self.learning_steps_bc} | loss/actor {round(loss_actor.item(), 3)} |')
            wandb.log({
                'PI/actor_loss': loss_actor.item(),
                })
            
    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )
    def load_models(self, save_dir):
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, 'actor.pth')))
        self.actor.to(self.device)
        print("Load model weight") 
