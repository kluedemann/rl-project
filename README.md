# RL Project: SAC
### By: Kai Lüdemann, Team: AGI_complete_final_3

This repository contains code for a soft actor-critic agent for the reinforcement learning final project.

## Overview
- `agents` - selected checkpoints, parameters and logs
- `checkpoint1` - test environments Pendulum-v1, HalfCheetah-v5
- `checkpoint2` - Hockey environment training modes
- `checkpoint3` - Hockey normal mode vs. basic/strong opponent
- `checkpoint4` - Hockey self-play learning
- `client` - Competition server client scripts
- `plots` - Clean plots for report
- `sac` - Package implementing soft actor-critic and training/running utilities 

## Setup

1. Clone GitHub repo
2. Install requirements
```
python3 -m venv venv
pip install -r requirements.txt
```
3. Install `sac` package locally from repository directory. 
```
pip install -e .
``` 

## Run agent client

```
python3 client/run_client.py <server args> --args --agent=sac
```


## Loading Trained Agent

```
from sac.utils import get_trained_agent
import pickle

# Load params
with open(path_to_params, "rb") as params_file:
    params = pickle.load(params_file)

# Get agent
agent = get_trained_agent(path_to_file, params)

# Sample actions
action = agent.act(obs, noise_scale=0.)
```