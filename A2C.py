from Network import *
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.algos.pg.a2c import A2C
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from EnvWrapper import rlpyt_make
import Config as C
import pandas as pd
import os
import os.path as osp

def findOptimalAgent (reward, run_ID=0) : 
    cpus = list(range(C.N_PARALLEL))
    affinity = dict(cuda_idx=C.CUDA_IDX, workers_cpus=cpus)
    sampler = SerialSampler(
        EnvCls=rlpyt_make,
        env_kwargs=dict(id=C.ENV, reward=reward),
        batch_T=C.BATCH_T,  
        batch_B=C.BATCH_B,  
        max_decorrelation_steps=400,
        eval_env_kwargs=dict(id=C.ENV),
        eval_n_envs=5,
        eval_max_steps=2500
    )
    algo = A2C(
        discount=C.DISCOUNT,
        learning_rate=C.LR,
        value_loss_coeff=C.VALUE_LOSS_COEFF,
        entropy_loss_coeff=C.ENTROPY_LOSS_COEFF
    )  
    agent = CategoricalPgAgent(AcrobotNet)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=C.N_STEPS,
        log_interval_steps=C.LOG_STEP,
        affinity=affinity,
    )
    name = "a2c_" + C.ENV.lower()
    log_dir = name
    with logger_context(log_dir, run_ID, name,
                        snapshot_mode='last', override_prefix=True):
        runner.train()
    return agent
