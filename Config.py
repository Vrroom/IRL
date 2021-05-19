import gym
from AcrobotUtils import toInternalStateRep

ENV                     = 'Acrobot-v1'
DISCOUNT                = 0.99
LR                      = 0.001
VALUE_LOSS_COEFF        = 0.5
ENTROPY_LOSS_COEFF      = 0.01
BATCH_T                 = 20
BATCH_B                 = 16
HDIMS                   = [512]
TRAJ_DIR                = './Trajectories'
IRL_ITR                 = 5
STATE_SIMILARITY_THRESH = 1e-1
IRL_STATE_SAMPLES       = 10
HORIZON                 = 500
L1_NORM_GAMMA           = 1
N_STEPS                 = 5e5
LOG_STEP                = 1e5
SEED                    = 1
INTERNAL_STATE_FN       = toInternalStateRep

with gym.make(ENV) as env : 
    ACTION_SPACE       = env.action_space.n
    OBSERVATION_SPACE  = env.observation_space.shape[0]

N_PARALLEL = 8
CUDA_IDX   = None
