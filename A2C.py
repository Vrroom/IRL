from Network import *
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.pg.a2c import A2C
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
import Config as C
import pandas as pd
import os
import os.path as osp
from bayes_opt import BayesianOptimization
from bayes_opt.event import DEFAULT_EVENTS, Events
from bayes_opt.logger import JSONLogger

run_ID = 0

def build_and_train(discount, log_lr, vlc, elc, cuda_idx=None, n_parallel=4):
    global run_ID
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=C.ENV),
        eval_env_kwargs=dict(id=C.ENV),
        batch_T=C.BATCH_T,  
        batch_B=16,  # 16 parallel environments.
        max_decorrelation_steps=400,
        eval_n_envs=5,
        eval_max_steps=2500
    )
    algo = A2C(
        discount=discount, 
        learning_rate=10 ** log_lr,
        value_loss_coeff=vlc,
        entropy_loss_coeff=elc
    )  
    agent = CategoricalPgAgent(AcrobotNet)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e5,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    name = "a2c_acrobot"
    log_dir = "a2c_acrobot"
    with logger_context(log_dir, run_ID, name, snapshot_mode='all', override_prefix=True):
        runner.train()
    dataframe = pd.read_csv(osp.join(log_dir, f'run_{run_ID}', 'progress.csv'))
    return dataframe['DiscountedReturnAverage'][-5:].mean()

def increment_run_id (event, instance) : 
    global run_ID
    run_ID += 1

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    # parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    # parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=4)
    # args = parser.parse_args()
    # print(build_and_train(
    #     run_ID=args.run_ID,
    #     cuda_idx=args.cuda_idx,
    #     n_parallel=args.n_parallel,
    # ))
    pbounds = dict(
        discount=(0.7, 1),
        log_lr=(-6, 0),
        vlc=(0, 10),
        elc=(0, 1)
    )
    optimizer = BayesianOptimization(
        f=build_and_train, 
        pbounds=pbounds,
        verbose=2,
        random_state=1
    )
    optimizer.subscribe(
        event=Events.OPTIMIZATION_STEP,
        subscriber='run id increment',
        callback=increment_run_id
    )
    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=2, n_iter=10)
