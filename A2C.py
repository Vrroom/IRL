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

def build_and_train(run_ID=0, cuda_idx=0, n_parallel=8):
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
    algo = A2C()  
    agent = CategoricalPgAgent(AcrobotNet)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    name = "a2c_" + C.ENV.lower()
    log_dir = name
    with logger_context(log_dir, run_ID, name, snapshot_mode='all', override_prefix=True):
        runner.train()

if __name__ == "__main__":
    build_and_train()
