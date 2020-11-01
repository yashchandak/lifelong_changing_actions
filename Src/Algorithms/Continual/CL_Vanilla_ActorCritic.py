import numpy as np
import torch
from torch import tensor, float32, ByteTensor
from torch.autograd import Variable
import torch.nn.functional as F
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, Policy, Critic, utils


class CL_Vanilla_ActorCritic(Agent):
    def __init__(self, config, action_mask):
        super(CL_Vanilla_ActorCritic, self).__init__(config)
        # Get state features and instances for Actor and Value function
        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = Policy.get_Policy(state_dim=self.state_features.feature_dim, config=config)
        self.critic = Critic.Critic(state_dim=self.state_features.feature_dim, config=config)
        self.trajectory = utils.Trajectory(max_len=self.config.batch_size, state_dim=self.state_dim,
                                           action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)

        self.modules = [('actor', self.actor), ('baseline', self.critic), ('state_features', self.state_features)]

        self.init()
        self.update_mask(action_mask=action_mask)

    def update_mask(self, action_mask):
        self.action_mask = Variable(torch.from_numpy(action_mask*1.0).type(ByteTensor), requires_grad=False)
        self.curr_action_set = np.where(action_mask)[0]

    # Overrides the reset function in parent class
    def reset(self, action_mask, change_flag):
        for _, module in self.modules:
           module.reset()

        if change_flag:
            if self.config.re_init == 'full':
                # Do a complete re initialization after the MDP has changed
                self.__init__(self.config, action_mask)
            if self.config.re_init == 'policy':
                # Re-init only the policy, (state features and value functions can carry over from prv time)
                self.actor, self.atype, self.action_size = Policy.get_Policy(state_dim=self.state_features.feature_dim,
                                                                             config=self.config)
            if self.config.re_init == 'none':
                # Do not re-init anything. Just unmask new set of parameters.
                pass

            self.update_mask(action_mask)

    def get_action(self, state, explore=0):
        explore = 0 # Don't do eps-greedy with policy gradients
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features.forward(state.view(1, -1))
        action, dist = self.actor.get_action(state, explore=explore, mask=self.action_mask)

        if self.config.debug:
            self.track_entropy(dist, action)
        return action, dist

    def update(self, s1, a1, dist, r1, s2, done):
        # Batch episode history, # Dont use value predicted from the absorbing/goal state
        self.trajectory.add(s1, a1, -1, r1, s2, int(done != 1))  # set dist=-1; Unused value
        # Equivalent to vanilla Actor Critic when batch_size = 1
        if self.trajectory.size >= self.config.batch_size or done:
            self.optimize()
            self.trajectory.reset()

    def optimize(self):
        s1, a1, _, r1, s2, not_absorbing = self.trajectory.get_all()

        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        # ---------------------- optimize critic ----------------------
        next_val = self.critic.forward(s2).detach()    # Detach targets from grad computation.
        val_exp  = r1 + self.config.gamma * next_val * not_absorbing
        val_pred = self.critic.forward(s1)

        # loss_critic = F.smooth_l1_loss(val_pred, val_exp)
        loss_critic = F.mse_loss(val_pred, val_exp)

        # ---------------------- optimize actor ----------------------
        td_error = (val_exp - val_pred).detach()
        logp, _ = self.actor.get_log_prob(s1, a1, mask=self.action_mask)
        loss_actor = -1.0 * torch.mean(td_error * logp)

        # print(s1.shape, next_val.shape, val_exp.shape, val_pred.shape, td_error.shape, self.actor.get_log_prob(s1, a1).shape)
        self.step(loss_actor + loss_critic, clip_norm=1)

