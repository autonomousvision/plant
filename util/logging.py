import json
import os
import glob
import subprocess
import argparse
import logging

from git import Repo
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime


def setup_logging(cfg):
    # Log args
    Path(cfg.model.training.ckpt_path).mkdir(parents=True, exist_ok=True)
    arg_dict = OmegaConf.to_container(cfg, resolve=True)
    args = argparse.Namespace(**arg_dict)
    with open(os.path.join(cfg.model.training.ckpt_path, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Log git
    sha = (
        subprocess.check_output(
            ["git", "-C", f"{cfg.user.working_dir}", "rev-parse", "HEAD"]
        )
        .decode("ascii")
        .strip()
    )
    commit = (
        subprocess.check_output(["git", "-C", f"{cfg.user.working_dir}", "log", "-1"])
        .decode("ascii")
        .strip()
    )
    branch = (
        subprocess.check_output(["git", "-C", f"{cfg.user.working_dir}", "branch"])
        .decode("ascii")
        .strip()
    )
    repo = Repo(cfg.user.working_dir)

    with open(os.path.join(cfg.model.training.ckpt_path, "git_info.txt"), "w") as f:
        # write current date and time
        f.write(
            f"Run started at: {str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))}\n"
        )
        f.write(f"Git state: {sha}\n")
        f.write(f"Git commit: {commit}\n")
        f.write(f"Git branch: {branch}\n\n")
        f.write(f"{repo.git.diff('HEAD')}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def sync_wandb(cfg):
    # TODO: sync wandb - still not working correctly
    wandb_files = glob.glob(f"./wandb/offline*/*.wandb")
    os.environ["TMPDIR"] = "/home/geiger/krenz73/tmp"
    for wandb_file in wandb_files:
        if os.path.getsize(wandb_file) > 5000000:
            os.system(f"wandb sync {wandb_file}")
