# RL Project: SAC
### By: Kai Lüdemann

## Loading Trained Agent

1. Clone GitHub repo
2. Install package locally using `pip install -e .` from repository directory.
3. Load trained agent from `.pth` file
```
from sac.utils import get_trained_agent
import pickle

with open(path_to_params, "rb") as params_file:
    params = pickle.load(params_file)

agent = get_trained_agent(path_to_file, params)
agent.act(obs, noise_scale=0.)
```