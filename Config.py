import gym

ENV                     = 'Acrobot-v1'
DISCOUNT                = 0.99
LR                      = 0.001
VALUE_LOSS_COEFF        = 0.5
ENTROPY_LOSS_COEFF      = 0.01
BATCH_T                 = 20
HDIMS                   = [512]
TRAJ_DIR                = './Trajectories'
IRL_ITR                 = 100
STATE_SIMILARITY_THRESH = 1e-1
IRL_STATE_SAMPLES       = 100
IRL_MONTE_CARLO_SAMPLES = 10
L1_NORM_GAMMA           = 1.0
N_STEPS                 = 1e6
LOG_STEP                = 1e5

with gym.make(ENV) as env : 
    ACTION_SPACE       = env.action_space.n
    OBSERVATION_SPACE  = env.observation_space.shape[0]
