# PlanT: Explainable Planning Transformers via Object-Level Representations

## [Paper](https://www.katrinrenz.de/plant/resources/2022_PlanT_CoRL.pdf) | [Project Page](http://www.katrinrenz.de/plant)


This repository provides code for the following paper:

- [Katrin Renz](https://www.katrinrenz.de), [Kashyap Chitta](https://kashyap7x.github.io/), [Otniel-Bogdan Mercea](https://merceaotniel.github.io/), [A. Sophia Koepke](https://www.eml-unitue.de/people/almut-sophia-koepke), [Zeynep Akata](https://www.eml-unitue.de/people/zeynep-akata) and [Andreas Geiger](http://www.cvlibs.net/),
*PlanT: Explainable Planning Transformers via Object-Level Representations*, CoRL 2022.  


![demo](gfx/plant_teaser2.gif)

# Content
* [ToDos](#todos)
* [Setup](#setup)
* [Data and models](#data-and-models)
* [Evaluation](#evaluation)
* [Citation](#citation)

## ToDos
- [x] Best checkpoint + evaluation
- [ ] Other checkpoints
- [ ] Dataset and training
- [ ] Data generation
- [ ] PlanT with perception
- [ ] Explainability metric


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
```

## Data and models
You can download our pretrained PlanT-medium 3x by executing:
``` bash
chmod +x download.sh
./download.sh
```

The other checkpoints and the dataset used for training will be uploaded soon.

## Evaluation
Start a Carla server:
```
# with display
./carla/CarlaUE4.sh --world-port=2000 -opengl
```
```
# without display
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./carla/CarlaUE4.sh --world-port=2000 -opengl
```

When the server is running, start the evaluation with:
```
python leaderboard/scripts/run_evaluation.py user=$USER experiments=PlanTmedium3x eval=longest6
```
You can find the results of the evaluation in a newly created evaluation folder inside the model folder. If you want to have a (very minimalistic) visualization you can set the `debug` flag (i.e., `python leaderboard/scripts/run_evaluation.py user=$USER experiments=PlanTmedium3x eval=longest6 debug=1`)

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

