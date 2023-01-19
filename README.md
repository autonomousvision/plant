# PlanT: Explainable Planning Transformers via Object-Level Representations


**News**: \
**19.01.2023:** We released the code to generate the attention visualization. \
**02.12.2022:** We released the perception checkpoint and the code for the SENSORS and MAP track agent. Conda environment needs to be updated. Checkpoints of the perception are in the checkpoint folder. Please download again. \
**11.11.2022:** We made some changes in the agent files to ensure compatibility with our perception PlanT. We therefore uploaded new checkpoint files. The old one does not work anymore with the current code.

## [Project Page](http://www.katrinrenz.de/plant) | [Paper](https://arxiv.org/abs/2210.14222) | [Supplementary](https://www.katrinrenz.de/plant/resources/PlanT_supp.pdf) 


This repository provides code for the following paper:

- [Katrin Renz](https://www.katrinrenz.de), [Kashyap Chitta](https://kashyap7x.github.io/), [Otniel-Bogdan Mercea](https://merceaotniel.github.io/), [A. Sophia Koepke](https://www.eml-unitue.de/people/almut-sophia-koepke), [Zeynep Akata](https://www.eml-unitue.de/people/zeynep-akata) and [Andreas Geiger](http://www.cvlibs.net/),
*PlanT: Explainable Planning Transformers via Object-Level Representations*, CoRL 2022.  


![demo](gfx/plant_teaser.gif)

# Content
* [Setup](#setup)
* [Data and models](#data-and-models)
* [Data generation](#data-generation)
* [Training](#training)
* [Evaluation](#evaluation)
* [Explainability](#explainability)
* [Perception PlanT](#perception-plant)
* [Citation](#citation)


## Setup
First, you have to install carla and the conda environment.

``` bash
# 1. Clone this repository
git clone https://github.com/autonomousvision/plant.git
cd plant
# 2. Setup Carla
# if you have carla already installed, skip the next step AND
# adapt the carla path in setup_env.sh before executing step 3.
chmod +x setup_carla.sh
./setup_carla.sh
# 3. Setup conda environment
chmod +x setup_env.sh
./setup_env.sh

conda activate plant
pip install -U openmim
mim install mmcv-full==1.7.0
pip install mmdet
```


## Data and models
You can download our pretrained PlanT models by executing:
``` bash
chmod +x download.sh
./download.sh
```

To download our 3x dataset run:
``` bash
chmod +x download_data.sh
./download_data.sh
```


## Data generation
You can download our dataset or generate your own dataset.
In order to generate your own one you first need to start a Carla server:
```
# with display
./carla/CarlaUE4.sh --world-port=2000 -opengl
```
```
# without display
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./carla/CarlaUE4.sh --world-port=2000 -opengl
```

To generate the data for the route specified in `carla_agent_files/config/eval/train.yaml` you can run
```
python leaderboard/scripts/run_evaluation.py user=$USER experiments=datagen eval=train
```
If you want to also save the sensor data that we used to train the perception module you can add the flag `experiments.SAVE_SENSORS=1`.

To generate the whole dataset you can use the `datagen.sh` file.


## Training
To run the PlanT training on the 3x dataset, run:
```
python training/PlanT/lit_train.py user=$USER model=PlanT
```
To change any hyperparameters have a look at `training/config/model/PlanT.yaml`. For general training settings (e.g., #GPUs) check `training/config/config.yaml`.


## Evaluation
This evaluates the PlanT model on the specified benchmark (default: longest6). The config is specified in the folder `carla_agent_files/config`.

Start a Carla server (see [Data generation](#data-generation)).\
When the server is running, start the evaluation with:
```
python leaderboard/scripts/run_evaluation.py user=$USER experiments=PlanTmedium3x eval=longest6
```
You can find the results of the evaluation in a newly created evaluation folder inside the model folder. If you want to have a (very minimalistic) visualization you can set the `viz` flag (i.e., `python leaderboard/scripts/run_evaluation.py user=$USER experiments=PlanTmedium3x eval=longest6 viz=1`)


## Explainability
The execution of the explainability agent contains two stages: (1) PlanT forwardpass (no execution of actions) to get attention weights. We filter the vehicles so that only the vehicles with the `topk` attention scores remain as input for the second step. (2) We execute either the expert or PlanT with the filtered input (the agent only sees `topk` vehicles instead of all).

Start a Carla server (see [Data generation](#data-generation)). \
When the server is running, start the evaluation with:
```
python leaderboard/scripts/run_evaluation.py user=$USER experiments=PlanTExplainability experiments.exec_model=Expert experiments.topk=1
```

To obtain the **attention visualization** set `experiments.topk=100000` and in addition add the flag `save_explainability_viz=True`. This saves a video per route in a `viz_vid` folder. The image resolution can be changed in `carla_agent_files/explainability_agent.py`. \
*Attention:* saving the videos slows the evaluation down.


## Perception PlanT
We release two PlanT agents suitable for the two CARLA Leaderboard tracks. For the SENSORS track we predict the route with our perception module. In the MAP track model we get the route information from the map. The code is taken from the [TransFuser (PAMI 2022) repo](https://github.com/autonomousvision/transfuser) and adapted for our usecase. The config is specified in the folder `carla_agent_files/config`. The config for the perception model is in `training/Perception/config.py`.

### SENSORS track
Start a Carla server (see [Data generation](#data-generation)).

When the server is running, start the evaluation with:
```
python leaderboard/scripts/run_evaluation.py user=$USER experiments=PlanTSubmission track=SENSORS eval=longest6 save_path=SENSORSagent
```
Visualization can be activated with the `viz` flag, and the unblocking from the TransFuser repo can be activated with the `experiments.unblock` flag.

### MAP track
Start a Carla server (see [Data generation](#data-generation)).

When the server is running, start the evaluation with:
```
python leaderboard/scripts/run_evaluation.py user=$USER experiments=PlanTSubmissionMap track=MAP eval=longest6 save_path=MAPagent
```
Visualization can be activated with the `viz` flag, and the unblocking from the TransFuser repo can be activated with the `experiments.unblock` flag.

## Citation
If you use this code and data, please cite the following:

```bibtex
@inproceedings{Renz2022CORL,
    author       = {Katrin Renz and Kashyap Chitta and Otniel-Bogdan Mercea and A. Sophia Koepke and Zeynep Akata and Andreas Geiger},
    title        = {PlanT: Explainable Planning Transformers via Object-Level Representations},
    booktitle    = {Conference on Robotic Learning (CoRL)},
    year         = {2022}
}
```

Also, check out the code for other recent work on CARLA from our group:
- [Chitta et al., TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving (PAMI 2022)](https://github.com/autonomousvision/transfuser)
- [Hanselmann et al., KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients (ECCV 2022)](https://github.com/autonomousvision/king)
- [Chitta et al., NEAT: Neural Attention Fields for End-to-End Autonomous Driving (ICCV 2021)](https://github.com/autonomousvision/neat)

