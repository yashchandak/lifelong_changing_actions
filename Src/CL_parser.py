import argparse
from datetime import datetime

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Parameters for Hyper-param sweep
        parser.add_argument("--base", default=-2, help="Base counter for Hyper-param search", type=int)
        parser.add_argument("--inc", default=-2, help="Increment counter for Hyper-param search", type=int)
        parser.add_argument("--hyper", default=-2, help="Which Hyper param settings", type=int)
        parser.add_argument("--seed", default=12345, help="seed for variance testing", type=int)

        # General parameters
        parser.add_argument("--save_count", default=1000, help="Number of ckpts for saving results and model", type=int)
        parser.add_argument("--optim", default='sgd', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",
                            choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=True, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--restore", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model ckpts")
        parser.add_argument("--summary", default=True, type=self.str2bool,
                            help="--UNUSED-- Visual summary of various stats")
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)

        # Book-keeping parameters
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")
        parser.add_argument("--experiment", default='Test_run', help="Name of the experiment")

        self.Env_n_Agent_args(parser)  # Decide the Environment and the Agent
        self.Main_AC_args(parser)  # General Basis, Policy, Critic
        self.CL_args(parser)  # Settings for Continual Learning
        self.parser = parser

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser

    def Env_n_Agent_args(self, parser):
        # parser.add_argument("--algo_name", default='CL_Vanilla_ActorCritic', help="")
        parser.add_argument("--algo_name", default='CL_DPG', help="")
        # parser.add_argument("--algo_name", default='CL_ActorCritic', help="Learning algorithm")
        parser.add_argument("--env_name", default='Gridworld_CL', help="Environment to run the code")
        parser.add_argument("--n_actions", default=8, help="number of base actions for gridworld", type=int)

        parser.add_argument("--max_episodes", default=int(5e4), help="maximum number of episodes (75000)", type=int)
        parser.add_argument("--max_steps", default=150, help="maximum steps per episode (500)", type=int)


    def CL_args(self, parser):
        parser.add_argument("--re_init", default='none', help="(none, policy, full) Reinitialize parameters on change")
        parser.add_argument("--freeze_action_rep", default=False, help="Freeze prv action rep on change",
                            type=self.str2bool)  # UNUSED
        parser.add_argument("--change_count", default=5, help="Number of CL changes", type=int)

        parser.add_argument("--valid_fraction", default=0.2, help="Fraction of data used for validation", type=float)
        parser.add_argument("--true_embeddings", default=False, help="Use ground truth embeddings or not?",
                            type=self.str2bool)
        parser.add_argument("--only_phase_one", default=False, help="Only phase1 training", type=self.str2bool)
        parser.add_argument("--emb_lambda", default=0, help="Lagrangian for learning embedding on the fly", type=float)
        parser.add_argument("--embed_lr", default=1e-4, help="Learning rate of action embeddings", type=float)
        parser.add_argument("--emb_reg", default=1e-2, help="L2 regularization for embeddings", type=float)
        parser.add_argument("--beta_vae", default=1e-2, help="Lagrangian for KL penalty", type=float)
        parser.add_argument("--emb_fraction", default=1, help="--UNUSED-- fraction of embeddings to consider",
                            type=float)
        parser.add_argument("--reduced_action_dim", default=2, help="dimensions of action embeddings", type=int)
        parser.add_argument("--load_embed", default=False, type=self.str2bool, help="Retrain flag")

        parser.add_argument("--sup_batch_size", default=16, help="(64)Supervised learning Batch size", type=int)
        parser.add_argument("--initial_phase_epochs", default=50, help="maximum number of episodes (150)", type=int)


    def Main_AC_args(self, parser):
        parser.add_argument("--exp", default=0.999, help="Eps-greedy epxloration decay", type=float)
        parser.add_argument("--gamma", default=0.99, help="Discounting factor", type=float)
        parser.add_argument("--trace_lambda", default=0.9, help="Lambda returns", type=float)
        parser.add_argument("--actor_lr", default= 0.00855, help="Learning rate of actor", type=float)
        parser.add_argument("--critic_lr", default= 0.01622, help="Learning rate of critic/baseline", type=float)
        parser.add_argument("--state_lr", default=1e-3, help="Learning rate of state features", type=float)
        parser.add_argument("--gauss_variance", default=1, help="Variance for gaussian policy", type=float)
        parser.add_argument("--entropy_lambda", default=0.01, help="Lagrangian for policy's entropy", type=float)
        parser.add_argument("--tau", default=0.001, help="soft update regularizer", type=float)

        parser.add_argument("--fourier_coupled", default=True, help="Coupled or uncoupled fourier basis",
                            type=self.str2bool)
        parser.add_argument("--fourier_order", default=3, help="Order of fourier basis, " +
                                                               "(if > 0, it overrides neural nets)", type=int)
        parser.add_argument("--NN_basis_dim", default='128', help="Shared Dimensions for Neural network layers")
        parser.add_argument("--Policy_basis_dim", default='2,16',
                            help="Dimensions for Neural network layers for policy")

        parser.add_argument("--buffer_size", default=75000, help="Size of memory buffer (3e5)", type=int)
        parser.add_argument("--batch_size", default=1, help="Batch size", type=int)
