import numpy as np
import torch
from torch import float32, ByteTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.utils import NeuralNet, pairwise_distances
from Src.Utils import Basis


class VAE_Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 config):
        super(VAE_Action_representation, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.norm_const = np.log(self.action_dim)
        self.action_mask = Variable(torch.from_numpy(np.zeros(self.action_dim)).type(ByteTensor), requires_grad=False)

        # Action embeddings to project the predicted action into original dimensions
        if config.true_embeddings:
            embeddings = config.env.get_embeddings() #motions.copy()
            self.reduced_action_dim = np.shape(embeddings)[1]
            maxi, mini = np.max(embeddings), np.min(embeddings)
            embeddings = ((embeddings - mini)/(maxi-mini))*2 - 1  # Normalize to (-1, 1)

            self.embeddings = Variable(torch.from_numpy(embeddings).type(float32), requires_grad=False)
        else:
            self.reduced_action_dim = config.reduced_action_dim
            if self.config.load_embed and self.config.re_init == 'none':
                try:
                    init_tensor = torch.load(self.config.paths['embedding'])['embeddings']
                except KeyError:
                    init_tensor = torch.load(self.config.paths['embedding'])['embeddings.weight']
                assert init_tensor.shape == (self.action_dim, self.reduced_action_dim)
                print("embeddings successfully loaded from: ", self.config.paths['embedding'])
            else:
                init_tensor = torch.rand(self.action_dim, self.reduced_action_dim)*2 - 1   # Don't initialize near the extremes.

            self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)

        # One layer neural net to get action representation
        self.fc_mean = nn.Linear(self.state_dim*2, self.reduced_action_dim)
        self.fc_std = nn.Linear(self.state_dim*2, self.reduced_action_dim)

        print("Action representation: ", [(name, param.shape) for name, param in self.named_parameters()])
        self.optim = config.optim(self.parameters(), lr=self.config.embed_lr)

    def update_mask(self, action_mask):
        self.action_mask = Variable(torch.from_numpy(action_mask*1.0).type(ByteTensor), requires_grad=False)
        self.norm_const = np.log(sum(action_mask))

    def get_match_scores(self, action):
        # compute similarity probability based on L2 norm
        embeddings = self.embeddings
        if not self.config.true_embeddings:
            embeddings = torch.tanh(embeddings)

        # compute similarity probability based on L2 norm
        similarity = - pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score

        # compute similarity probability based on dot product
        # similarity = torch.mm(action, torch.transpose(embeddings, 0, 1))  # Dot product

        # Never choose the actions not in the active set
        # Negative infinity ensures that these actions probability evaluates to 0 (e^-inf) during softmax as well
        similarity[:, self.action_mask == False] = float('-inf')  # Dimension = (bacth_size x unmasked number of actions)

        return similarity

    def get_best_match(self, action):
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1)
        return pos.cpu().item() #data.numpy()[0]

    def get_embedding(self, action):
        # Get the corresponding target embedding
        action_emb = self.embeddings[action]
        if not self.config.true_embeddings:
            action_emb = torch.tanh(action_emb)
        return action_emb

    def forward(self, state1, state2):
        # concatenate the state features and predict the action required to go from state1 to state2
        state_cat = torch.cat([state1, state2], dim=1)
        mu = self.fc_mean(state_cat)
        log_var = self.fc_std(state_cat)

        std = log_var.div(2).exp()
        eps = torch.randn_like(std)
        x = mu + std*eps

        # Bound the output of the encoder network.
        x = torch.tanh(x)

        return x, mu, log_var

    def unsupervised_loss(self, s1, a, s2, normalized=True):
        x, mu, log_var = self.forward(s1, s2)
        similarity = self.get_match_scores(x)  # Negative euclidean
        klds = -0.5*(1 + log_var - mu.pow(2) - log_var.exp())
        if normalized:
            loss = F.cross_entropy(similarity, a, reduction='mean')/self.norm_const \
                   + self.config.emb_reg * torch.pow(self.embeddings, 2).mean()/(1.0 * self.reduced_action_dim) \
                   + self.config.beta_vae * klds.mean()
                   # + self.config.emb_reg * torch.pow(self.embeddings[self.action_mask], 2).mean()/self.reduced_action_dim
        else:
            loss = F.cross_entropy(similarity, a, reduction='mean') \
                   + self.config.emb_reg * torch.pow(self.embeddings, 2).mean()
                   # + self.config.emb_reg * torch.pow(self.embeddings[self.action_mask], 2).mean()
        return loss


