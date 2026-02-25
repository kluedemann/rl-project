#!/bin/bash

export COMPRL_SERVER_URL=comprl.cs.uni-tuebingen.de
export COMPRL_SERVER_PORT=65335
export COMPRL_ACCESS_TOKEN=<your token>

chmod +x ./venv/bin/activate
./venv/bin/activate
./autorestart.sh --server-url $COMPRL_SERVER_URL --server-port $COMPRL_SERVER_PORT --token $COMPRL_ACCESS_TOKEN --args --agent=sac

