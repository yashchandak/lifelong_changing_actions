import numpy as np
import torch
from torch.autograd import Variable
from torch import tensor, float32, long
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.utils import NeuralNet, MemoryBuffer, OrnsteinUhlenbeckActionNoise, soft_update, hard_update
from copy import deepcopy
from collections import deque
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, Policy, Critic
from Src.Algorithms.Continual import CL_ActionRepresentation

# from memory_profiler import profile
class Actor(NeuralNet):
    def __init__(self, action_dim, state_dim, config):
        super(Actor, self).__init__()

        # Initialize network architecture and optimizer
        self.fc1 = nn.Linear(state_dim, action_dim)
        self.custom_weight_init()
        print("Actor: ", [(m, param.shape) for m, param in self.named_parameters()])
        self.optim = config.optim(self.parameters(), lr=config.actor_lr)

    def get_action(self, state):
        # Output the action embedding
        action = torch.tanh(self.fc1(state))
        return action

class Q_fn(NeuralNet):
    def __init__(self, action_dim, state_dim, config):
        super(Q_fn, self).__init__()

        self.fc1 = nn.Linear(action_dim + state_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.custom_weight_init()
        print("Critic: ", [(m, param.shape) for m, param in self.named_parameters()])
        self.optim = config.optim(self.parameters(), lr=config.critic_lr)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class CL_DPG(Agent):
    # @profile
    def __init__(self, config, action_mask):
        super(CL_DPG, self).__init__(config)

        # Set Hyper-parameters

        self.initial_phase = not config.true_embeddings and not config.load_embed and not config.restore # Initial training phase required if learning embeddings
        self.batch_norm = False

        # Function to get state features and action representation
        self.state_features = Basis.get_Basis(config=config)
        self.action_rep = CL_ActionRepresentation.VAE_Action_representation(action_dim=self.action_dim, state_dim=self.state_features.feature_dim, config=config)
        # Create instances for Actor and Q_fn
        self.actor = Actor(action_dim=self.action_rep.reduced_action_dim, state_dim=self.state_features.feature_dim, config=config)
        self.Q = Q_fn(action_dim=self.action_rep.reduced_action_dim, state_dim=self.state_features.feature_dim, config=config)

        # Create target networks
        # Deepcopy not working.
        self.target_state_features = Basis.get_Basis(config=config)
        self.target_actor = Actor(action_dim=self.action_rep.reduced_action_dim, state_dim=self.state_features.feature_dim, config=config)
        self.target_Q = Q_fn(action_dim=self.action_rep.reduced_action_dim,state_dim=self.state_features.feature_dim, config=config)
        # self.target_action_rep = ActionRepresentation.Action_representation_deep(action_dim=self.action_dim, config=config)
        # Copy the initialized values to target
        self.target_state_features.load_state_dict(self.state_features.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_Q.load_state_dict(self.Q.state_dict())
        # self.target_action_rep.load_state_dict(self.action_rep.state_dict())



        self.memory = MemoryBuffer(max_len=self.config.buffer_size, state_dim=self.state_dim,
                                     action_dim=1, atype=long, config=config,
                                     dist_dim=self.action_rep.reduced_action_dim)  # off-policy
        self.noise = OrnsteinUhlenbeckActionNoise(self.config.reduced_action_dim)


        self.modules = [('actor', self.actor), ('Q', self.Q), ('state_features', self.state_features), ('action_rep', self.action_rep),
                        ('target_actor', self.target_actor), ('target_state_features', self.target_state_features), ('target_Q', self.target_Q)]#,
                        # ('target_action_rep', self.target_action_rep)]

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


            self.update_mask(action_mask)
            self.initial_phase = True
            self.memory.reset()


    def get_action(self, state, explore=0):
        if self.batch_norm: self.actor.eval()  # Set the actor to Evaluation mode. Required for Batchnorm

        if self.initial_phase:
            # take random actions (uniformly in actual action space) to observe the interactions initially
            action = np.random.choice(self.curr_action_set)
            action_emb = self.action_rep.get_embedding(action).cpu().view(-1).data.numpy()

        else:
            state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device).view(1, -1)
            state = self.state_features.forward(state)
            action_emb = self.actor.get_action(state)

            noise = self.noise.sample() * explore  #* 0.1
            action_emb += Variable(torch.from_numpy(noise).type(float32), requires_grad=False)

            action = self.action_rep.get_best_match(action_emb)
            action_emb = action_emb.cpu().view(-1).data.numpy()

        self.track_entropy_cont(action_emb)
        return action, action_emb

    def update(self, s1, a1, a_emb1, r1, s2, done):
        self.memory.add(s1, a1, a_emb1, r1, s2, int(done != 1))
        if self.initial_phase and self.memory.length >= self.config.buffer_size:
            self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
        elif not self.initial_phase and self.memory.length > self.config.sup_batch_size:
            self.optimize()

    def optimize(self):
        if self.batch_norm: self.actor.train()  # Set the actor to training mode. Required for Batchnorm

        s1, a1, a1_emb, r1, s2, not_absorbing = self.memory.sample(self.config.sup_batch_size)


        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        s2_t = self.target_state_features.forward(s2).detach()
        a2_emb = self.target_actor.get_action(s2_t).detach()                      # Detach targets from grad computation.
        next_val = self.target_Q.forward(s2_t, a2_emb).detach()                   # Compute Q'( s2, pi'(s2))
        val_exp  = r1 + self.config.gamma * next_val * not_absorbing           # y_exp = r + gamma * Q'( s2, pi'(s2))

        s1_ = self.state_features.forward(s1)
        val_pred = self.Q.forward(s1_, a1_emb)                   # y_pred = Q( s1, a1)
        loss_Q = F.mse_loss(val_pred, val_exp)
        # loss_Q = F.smooth_l1_loss(val_pred, val_exp)                    # compute critic loss

        self.clear_gradients()
        loss_Q.backward()
        self.Q.optim.step()
        self.state_features.optim.step()

        # ---------------------- optimize actor ----------------------
        s1_ = self.state_features.forward(s1)
        s2_ = self.state_features.forward(s2)
        pred_a1_emb = self.actor.get_action(s1_)
        loss_actor = -1.0 * torch.mean(self.Q.forward(s1_, pred_a1_emb))
        loss_rep = self.action_rep.unsupervised_loss(s1_, a1.view(-1), s2_) * self.config.emb_lambda

        loss = loss_actor + loss_rep
        self.clear_gradients()
        loss.backward()
        self.actor.optim.step()
        self.action_rep.optim.step()
        self.state_features.optim.step()

        # ------------ update target actor and critic -----------------
        soft_update(self.target_actor, self.actor, self.config.tau)
        soft_update(self.target_Q, self.Q, self.config.tau)
        soft_update(self.target_state_features, self.state_features, self.config.tau)

    def self_supervised_update(self, s1, a1, s2, reg=1):
        s1 = self.state_features(s1)
        s2 = self.state_features(s2)

        loss = self.action_rep.unsupervised_loss(s1, a1.view(-1), s2) * reg

        self.clear_gradients()
        loss.backward()
        self.action_rep.optim.step()
        self.state_features.optim.step()

        return loss.item()

    def clear_gradients(self):
        for module in [self.action_rep, self.actor, self.Q, self.state_features]:
            module.optim.zero_grad()

    def initial_phase_training(self, max_epochs=-1):
        if self.batch_norm: self.actor.train()  # Set the actor to training mode. Required for Batchnorm

       # change optimizer to Adam for unsupervised learning
        self.action_rep.optim = torch.optim.Adam(self.action_rep.parameters(), lr=1e-2)
        self.state_features.optim = torch.optim.Adam(self.state_features.parameters(), lr=1e-2)
        initial_losses = []

        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            for s1, a1, _, _, s2, _ in self.memory.batch_sample(batch_size=self.config.sup_batch_size,
                                                                randomize=True):
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

        hard_update(self.target_state_features, self.state_features)