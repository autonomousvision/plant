import argparse
import os
import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="../../carla_agent_files/config", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg_org = cfg.copy()
    cfg = cfg.experiments

    print(cfg_org.eval.routes)
    print(cfg_org.checkpoint)
    print("Working directory : {}".format(os.getcwd()))
    print(f'Save gifs: {cfg_org.save_explainability_viz}')

    # create result folder
    Path(cfg_org.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    
    os.environ["CARLA_SERVER"] = f"{cfg_org.user.carla_path}/CarlaUE4.sh"
    # sys.path.append(f"{cfg_org.user.carla_path}/PythonAPI")
    # sys.path.append(f"{cfg_org.user.carla_path}/PythonAPI/carla")
    # sys.path.append(f"{cfg_org.user.carla_path}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg")
    # sys.path.append(f"leaderboard")
    # sys.path.append(f"scenario_runner")
    

    # os.environ["SCENARIO_RUNNER_ROOT"] = f"scenario_runner"
    # os.environ["LEADERBOARD_ROOT"] = f"leaderboard"
    os.environ["AGENT_NAME"] = f"{cfg.name}"
    os.environ["DEBUG_CHALLENGE"] = f"{cfg_org.DEBUG_CHALLENGE}"
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{cfg_org.CUDA_VISIBLE_DEVICES}"
    os.environ["DATAGEN"]= f"{cfg.DATAGEN}"
    os.environ["SHUFFLE_WEATHER"]= f"{cfg.SHUFFLE_WEATHER}"
    os.environ["BENCHMARK"]= f"{cfg_org.eval.BENCHMARK}"
    os.environ["SAVE_GIF"]= f"{cfg_org.save_explainability_viz}"
    os.environ["LOG_SAVE_PATH"]= f"{cfg_org.log_save_path}"

    if int(cfg.DATAGEN) == 0:
        pass
        # os.environ["LOAD_CKPT_PATH"]= f"{cfg.model_ckpt_load_path}"
    else:
        os.environ["DATA_SAVE_PATH"]= f"{cfg.data_save_path}"

    arg_dict0 = OmegaConf.to_container(cfg_org.eval, resolve=True)
    arg_dict1 = OmegaConf.to_container(cfg, resolve=True)
    arg_dict2 = OmegaConf.to_container(cfg_org, resolve=True)
    arg_dict1.update(arg_dict2)
    arg_dict1.update(arg_dict0)
    args=argparse.Namespace(**arg_dict1)

    from leaderboard import leaderboard_evaluator_local
    leaderboard_evaluator_local.main_eval(args)


if __name__ == '__main__':
    main()