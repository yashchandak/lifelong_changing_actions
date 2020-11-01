import numpy as np
import torch
from torch.autograd import Variable
from torch import tensor, float32, long
import torch.nn.functional as F
from Src.Utils.utils import MemoryBuffer, Trajectory
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, Policy, Critic
from Src.Algorithms.Continual import CL_ActionRepresentation


class CL_ActorCritic(Agent):
    def __init__(self, config, action_mask):
        super(CL_ActorCritic, self).__init__(config)

        # Initial training phase required if learning embeddings from scratch
        self.initial_phase = not config.true_embeddings and not config.load_embed

        # Function to get state features and action representation
        self.state_features = Basis.get_Basis(config=config)
        self.action_rep = CL_ActionRepresentation.VAE_Action_representation(state_dim=self.state_features.feature_dim,
                                                                     action_dim=self.action_dim, config=config)

        # Create instances for Actor and Q_fn
        self.critic = Critic.Critic_with_traces(state_dim=self.state_features.feature_dim, config=config)
        self.actor = Policy.embed_Gaussian(action_dim=self.action_rep.reduced_action_dim,
                                           state_dim=self.state_features.feature_dim, config=config)

        # Initialize storage containers
        self.memory =   MemoryBuffer(max_len=self.config.buffer_size, state_dim=self.state_dim,
                                     action_dim=1, atype=long, config=config,
                                     dist_dim=self.action_rep.reduced_action_dim)  # off-policy
        self.trajectory = Trajectory(max_len=self.config.batch_size, state_dim=self.state_dim,
                                     action_dim=1, atype=long, config=config,
                                     dist_dim=self.action_rep.reduced_action_dim)  # on-policy

        self.modules = [('actor', self.actor), ('critic', self.critic),
                        ('state_features', self.state_features), ('action_rep', self.action_rep)]

        self.init()
        self.update_mask(action_mask=action_mask)

    def update_mask(self, action_mask):
        self.action_mask = action_mask
        self.curr_action_set = np.where(self.action_mask)[0]
        self.action_rep.update_mask(self.action_mask)

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
                self.action_rep = CL_ActionRepresentation.Action_representation(
                    state_dim=self.state_features.feature_dim,
                    action_dim=self.action_dim, config=self.config)
                self.actor = Policy.embed_Gaussian(action_dim=self.action_rep.reduced_action_dim,
                                                   state_dim=self.state_features.feature_dim, config=self.config)

            self.update_mask(action_mask)
            self.initial_phase = True
            self.memory.reset()

    def get_action(self, state, explore=0):
        explore = 0  # Don't do eps-greedy with policy gradients
        if self.initial_phase or np.random.rand() < explore:
            # take random actions (uniformly in actual action space) to observe the interactions initially
            action = np.random.choice(self.curr_action_set)
            chosen_action_emb = self.action_rep.get_embedding(action).cpu().view(-1).data.numpy()

        else:
            state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device).view(1, -1)
            state = self.state_features.forward(state)
            chosen_action_emb, _ = self.actor.get_action(state, explore=0)
            action = self.action_rep.get_best_match(chosen_action_emb)

            chosen_action_emb = chosen_action_emb.cpu().view(-1).data.numpy()

        return action, chosen_action_emb

    def update(self, s1, a1, a_emb1, r1, s2, done, debug=False):
        if not self.initial_phase:
            # On-policy episode history, # Dont use value predicted from the absorbing/goal state
            # self.optimize(s1, a1, a_emb1, r1, s2, int(done != 1))
            self.trajectory.add(s1, a1, a_emb1, r1, s2, int(done != 1))
            if self.trajectory.size >= self.config.batch_size or done:
                self.optimize(debug)
                self.trajectory.reset()
        else:
            # action embeddings can be learnt offline
            self.memory.add(s1, a1, a_emb1, r1, s2, int(done != 1))
            if self.memory.length >= self.config.buffer_size:
                self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)

    def optimize(self, debug=False):
        s1, a1, chosen_a1_emb, r1, s2, not_absorbing = self.trajectory.get_all()

        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        # ---------------------- optimize critic ----------------------
        next_val = self.critic.forward(s2).detach()    # Detach targets from grad computation.
        val_exp  = r1 + self.config.gamma * next_val * not_absorbing
        val_pred = self.critic.forward(s1)
        loss_critic = F.mse_loss(val_pred, val_exp)

        # loss_critic = F.smooth_l1_loss(val_pred, val_exp)

        # ---------------------- optimize actor ----------------------
        td_error = (val_exp - val_pred).detach()
        logp, dist = self.actor.get_log_prob(s1, chosen_a1_emb)
        loss_actor = -1.0 * torch.mean(td_error * logp)
        # loss_actor += self.config.entropy_lambda * self.actor.get_entropy_from_dist(dist)

        # Take one policy gradient step
        loss = loss_critic + loss_actor

        # if not self.config.true_embeddings and self.config.emb_lambda > 0:
        #    loss += self.action_rep.unsupervised_loss(s1, a1.view(-1), s2) * self.config.emb_lambda

        self.step(loss, clip_norm=1)


    def self_supervised_update(self, s1, a1, s2, reg=1):
        self.clear_gradients()  # clear all the gradients from last run

        # If doing online updates, sharing the state features might be problematic!
        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        # ------------ optimize the embeddings ----------------
        loss_act_rep = self.action_rep.unsupervised_loss(s1, a1.view(-1), s2, normalized=True) * reg
        loss_act_rep.backward()

        # Directly call the optimizer's step fn to bypass lambda traces (if any)
        self.action_rep.optim.step()
        self.state_features.optim.step()

        return loss_act_rep.item()

    def initial_phase_training(self, max_epochs=-1):
        # change optimizer to Adam for unsupervised learning
        self.action_rep.optim = torch.optim.Adam(self.action_rep.parameters(), lr=1e-2)
        self.state_features.optim = torch.optim.Adam(self.state_features.parameters(), lr=1e-2)
        initial_losses = []

        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            for s1, a1, _, _, s2, _ in self.memory.batch_sample(batch_size=self.config.sup_batch_size, randomize=True):
                loss = self.self_supervised_update(s1, a1, s2)
                losses.append(loss)

            initial_losses.append(np.mean(losses))
            if counter % 1 == 0:
                print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-10:])))
                if self.config.only_phase_one:
                    self.save()
                    print("Saved..")

            # Terminate initial phase once action representations have converged.
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) + 1e-5 >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break

        # Reset the optim to whatever is there in config
        self.action_rep.optim = self.config.optim(self.action_rep.parameters(), lr=self.config.embed_lr)
        self.state_features.optim = self.config.optim(self.state_features.parameters(), lr=self.config.state_lr)

        print('... Initial training phase terminated!')
        self.initial_phase = False
        self.save()

        if self.config.only_phase_one:
            exit()


