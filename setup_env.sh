conda env create -f environment.yml
# conda activate plant

pwd_root=$(pwd)

cd $CONDA_PREFIX/envs/plant
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

echo "export CARLA_ROOT=\${pwd_root}/carla" >> ./etc/conda/activate.d/env_vars.sh
echo "export CARLA_SERVER=\${CARLA_ROOT}/CarlaUE4.sh" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:leaderboard" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:scenario_runner" >> ./etc/conda/activate.d/env_vars.sh

echo "unset CARLA_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset CARLA_SERVER" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset PYTHONPATH" >> ./etc/conda/deactivate.d/env_vars.sh

cd $pwd_root

touch ./carla_agent_files/config/user/$USER.yaml
touch ./training/config/user/$USER.yaml
echo "working_dir: $pwd_root" >> ./carla_agent_files/config/user/$USER.yaml
echo "working_dir: $pwd_root" >> ./training/config/user/$USER.yaml
echo "carla_path: $pwd_root/carla" >> ./carla_agent_files/config/user/$USER.yaml

# conda deactivate
# conda activate plant