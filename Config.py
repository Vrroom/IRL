""" All constants in one place!  """
import gym
from AcrobotUtils import toInternalStateRep
import multiprocessing as mp

# Facts about the MDP
ENV                     = 'Acrobot-v1'
DISCOUNT                = 0.99
HORIZON                 = 500
INTERNAL_STATE_FN       = toInternalStateRep

# Trajectory sampling parameters
BATCH_T                 = 20
BATCH_B                 = 16

# Parameters for the IRL algorithm
IRL_ITR                 = 5     # number of optimization steps
L1_NORM_GAMMA           = 1     # weight of L1 Norm in LP
STATE_SIMILARITY_THRESH = 1e-1  
IRL_STATE_SAMPLES       = 10
TRAJ_DIR                = './Trajectories'

# Policy and Value neural network parameters
HDIMS                   = [512]
with gym.make(ENV) as env : 
    ACTION_SPACE        = env.action_space.n
    OBSERVATION_SPACE   = env.observation_space.shape[0]

# Number of workers to use for sampling 
# and cuda device id
N_PARALLEL              = mp.cpu_count()
CUDA_IDX                = None

# Constants for training an agent using A2C
LR                      = 0.001
VALUE_LOSS_COEFF        = 0.5
ENTROPY_LOSS_COEFF      = 0.01
N_STEPS                 = 5e5  # steps for training
LOG_STEP                = 1e5

# For all randomness!
SEED                    = 1
