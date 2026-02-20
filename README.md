# RL Project: SAC
### By: Kai Lüdemann

## Loading Trained Agent

1. Clone GitHub repo
2. Install package locally using `pip install -e .` from repository directory.
3. Load trained agent from `.pth` file
```
from sac.utils import get_trained_agent
agent = get_trained_agent(path_to_file)
o, info = env.reset()
agent.act(o)
```