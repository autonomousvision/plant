# CARLA path
# export CARLA_ROOT=/home/kchitta/Documents/CARLA_0.9.10.1
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:scenario_runner
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=MAP

# Server Ports
export PORT=2000 # same as the carla server port
export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients

# Evaluation Setup
export ROUTES=leaderboard/data/routes/val/val_mixed.xml
export SCENARIOS=leaderboard/data/scenarios/eval_scenarios.json
export DP_SEED=0 # seed for initializing the locations of background actors
export DEBUG_CHALLENGE=0 # visualization of waypoints and forecasting
export SHUFFLE_WEATHER=1 # whether to shuffle the weather each frame for data augmentation (NEAT eval: 0)
export RESUME=0
export REPETITIONS=1

# Agent Paths
export TEAM_AGENT=inference/brakeonly_agent.py # agent
export TEAM_CONFIG=tmp/ # model checkpoint
export CHECKPOINT_ENDPOINT=carla_results/temp.json # output results file
# export DATA_SAVE_PATH=../carla_data/hybrid_agent/val_mixed # path for saving episodes (comment to disable)

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--dataProviderSeed=${DP_SEED}

# python3 tools/generate_intersection_routes.py \
# --town=Town06 \
# --save_file=leaderboard/data/routes/new_town06.xml \

# python3 tools/sample_junctions.py \
# --routes_file=leaderboard/data/routes/new_town01.xml \
# --save_file=leaderboard/data/routes/new_town01_sampled.xml

# python3 webdataset/create_webdataset.py

# python3 tools/vis_xml.py \
# --xml_path=leaderboard/data/dense/right/routes_10mshortroutes_Town10HD_Scenario8junction.xml \
# --map=Town10HD