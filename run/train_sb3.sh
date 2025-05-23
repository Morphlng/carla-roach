#!/bin/bash

train_rl () {
  python -u train_sb3.py \
  wb_project=sb3_baselines wb_name=TD3 \
  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}

# To use gaussian distribution: `agent/ppo/policy=xtma_gaus`
# To disable exploration loss: `agent.ppo.training.kwargs.explore_coef=0.0`
# To resume a crashed run, set `agent.ppo.wb_run_path` to the w&b run path


# NO NEED TO MODIFY THE FOLLOWING
# actiate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carla

# resume benchmark in case carla is crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  train_rl
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now