import glob
import os
import json
import subprocess
import time
from pathlib import Path


def make_jobsub_file(evalFolder_name, root, model_path, agent, evaluation_benchmark, scenario_file, route_path, epoch, job_number,job_name):

    dataset = f'{evalFolder_name}/{evaluation_benchmark}/epoch{epoch}/slurm_{Path(route_path).stem}'
    os.makedirs(f"{root}/{model_path}/{dataset}/run_files/logs", exist_ok=True)
    os.makedirs(f"{root}/{model_path}/{dataset}/run_files/job_files", exist_ok=True)
    job_file = f"{root}/{model_path}/{dataset}/run_files/job_files/{job_number}.sh"
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{job_number}
#SBATCH --partition=gpu-2080ti
#SBATCH -o {root}/{model_path}/{dataset}/run_files/logs/qsub_out{job_number}.log
#SBATCH -e {root}/{model_path}/{dataset}/run_files/logs/qsub_err{job_number}.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=katrin.renz@uni-tuebingen.de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time=00-24:00
#SBATCH --gres=gpu:1
# -------------------------------

# print info about current job
scontrol show job $SLURM_JOB_ID 

port=`python util/get_carla_port.py`
TMport=$((port+8000))

echo "Port: $port"

SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 /home/geiger/krenz73/software/carla_0_9_10/CarlaUE4.sh --world-port=$port -opengl &

sleep 200
CUDA_VISIBLE_DEVICES=0 \
python leaderboard/scripts/run_evaluation.py \
experiments={agent} \
port=$port \
trafficManagerPort=$TMport \
save_gif=False \
experiments.model_path={model_path} \
'experiments.model_ckpt_load_path="{root}/{model_path}/checkpoints/epoch={epoch}.ckpt"' \
resume=0 \
route_rel={route_path} \
scenarios_rel={scenario_file} \
save_path={dataset} \
checkpoint_file={Path(route_path).stem}.json \
BENCHMARK={evaluation_benchmark} \
"""
    with open(job_file, "w") as f:
        f.write(qsub_template)
    return job_file


def get_num_jobs(job_name="datagen", username="krenz73"):
    # print(job_name)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:7,name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            # f"SQUEUE_FORMAT2='username:7,name:130' squeue --sort V | grep {username} | wc -l",
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", "")
    )
    max_num_parallel_jobs = int(open("max_num_jobs.txt", "r").read())
    return num_running_jobs, max_num_parallel_jobs



if __name__ == "__main__":

    evaluation_iterations = 1
    iter_offset = 0
    evalFolder_name_base = 'carla_eval'
    evaluation_benchmark = "longest6"
    agent = "PlanT" 
    root = '/home/geiger/krenz73/coding/02_sequential_driving/release/plant'
    job_name = 'evrelPlanT'
    run_jobs = False

    epochs = ['047']#, '029', '035', '047', '053']

    models = [
        'checkpoints/PlanT/3x/PlanT_medium',
        ]

    print(f'Found {len(models)} models')



    if evaluation_benchmark == "longest6":
        route_root = 'leaderboard/data/longest6/longest6_split'
        routes = glob.glob(os.path.join(route_root, '*.xml'))
        scenario_file = "leaderboard/data/longest6/eval_scenarios.json"
    
    elif evaluation_benchmark == "lav":
        route_root = 'leaderboard/data/lav/lav_split'
        routes = glob.glob(os.path.join(route_root, '*.xml'))
        scenario_file = "leaderboard/data/lav/all_towns_traffic_scenarios_public.json"

    elif evaluation_benchmark == "neat":
        route_root = 'leaderboard/data/neat/neat_split'
        routes = glob.glob(os.path.join(route_root, '*.xml'))
        scenario_file = "leaderboard/data/neat/eval_scenarios.json"
    else:
        raise NotImplementedError

    meta_jobs = {}

    job_number = 0
    tot_num_jobs = len(routes) * len(epochs) * len(models) * evaluation_iterations
    print(f'Total number of jobs: {tot_num_jobs}')


    if run_jobs:
        for epoch in epochs:
            for model in models:
                for iteration in range(evaluation_iterations):

                    print(model)
                

                    evalFolder_name = f'{evalFolder_name_base}_{iteration+iter_offset}'

                    if Path(f'{root}/{model}/{evalFolder_name}/{evaluation_benchmark}/epoch{epoch}').exists():
                        print(f'{evalFolder_name}/{evaluation_benchmark}/epoch{epoch} already exists')
                        print('Do you want to overrite? (y/n)')
                        if input() == 'y':
                            print('Ok continue starting.')
                        else:
                            print('Ok. Skipping!')
                            continue

                    if agent != "RuleBased_planner":
                        ckpt = f'{root}/{model}/checkpoints/epoch={epoch}.ckpt'

                        # if not os.path.exists(ckpt):
                        #     continue
                        
                        while not os.path.exists(ckpt):
                            print(f'Waiting for {ckpt} to be created')
                            try:
                                time.sleep(180)
                            except:
                                break
                    
                    

                    for route in routes:


                        route_path = route
                        # model_path = f'{root}/{model_path}'
                    
                        job_file = make_jobsub_file(evalFolder_name, root, model, agent, evaluation_benchmark, scenario_file, route_path, epoch, job_number, job_name)
                        result_file = f"{model}/{evalFolder_name}/{evaluation_benchmark}/epoch{epoch}/slurm_{Path(route_path).stem}/{Path(route_path).stem}.json"
                        
                        num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name)
                        print(f"{num_running_jobs}/{max_num_parallel_jobs} jobs are running...")
                        while num_running_jobs >= max_num_parallel_jobs:
                            num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name)
                            time.sleep(30)
                        print(f"Submitting job {job_number}/{tot_num_jobs}: {job_name}")

                        # os.system(f"sbatch {job_file}")
                        job_number += 1

                        jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8").strip().split(' ')[-1]
                        meta_jobs[jobid] = (False, job_file, result_file, 0)


                        time.sleep(0.2) # because of automatic carla port assignment


        time.sleep(380)
        training_finished = False
        while not training_finished:
            num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name)
            print(f"{num_running_jobs} jobs are running... Job: {job_name}")
            time.sleep(120)

            # resubmit unfinished jobs
            for k in list(meta_jobs.keys()):
                job_finished, job_file, result_file, resubmitted = meta_jobs[k]
                need_to_resubmit = False
                if not job_finished and resubmitted < 2:
                    # check whether job is runing
                    if int(subprocess.check_output(f'squeue | grep {k} | wc -l', shell=True).decode("utf-8").strip()) == 0:
                        # check whether result file is finished?
                        if os.path.exists(result_file):
                            evaluation_data = json.load(open(result_file))
                            progress = evaluation_data['_checkpoint']['progress']
                            if len(progress) < 2 or progress[0] < progress[1]:
                                # resubmit
                                need_to_resubmit = True
                            else:
                                for record in evaluation_data['_checkpoint']['records']:
                                    if(record['scores']['score_route'] <= 0.00000000001 ):
                                        need_to_resubmit = True
                                    if(record['status'] == "Failed - Agent couldn\'t be set up"):
                                        need_to_resubmit = True
                                    if(record['status'] == "Failed"):
                                        need_to_resubmit = True
                                    if(record['status'] == "Failed - Simulation crashed"):
                                        need_to_resubmit = True
                                    if(record['status'] == "Failed - Agent crashed"):
                                        need_to_resubmit = True
                            
                            if not need_to_resubmit:
                                # delete old job
                                print(f'Finished job {job_file}')
                                meta_jobs[k] = (True, None, None, 0)
                                
                        else:
                            need_to_resubmit = True

                if need_to_resubmit:
                    print(f"resubmit sbatch {job_file}")

                    # rename old error files to still access it
                    os.system(f"cp -r {Path(result_file).parent}/run_files/logs {Path(result_file).parent}/run_files/logs_{time.time()}")

                    jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8").strip().split(' ')[-1]
                    meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
                    meta_jobs[k] = (True, None, None, 0)
                    

            time.sleep(10)

            if num_running_jobs == 0:
                training_finished = True




    for iteration in range(evaluation_iterations):
        evalFolder_name = f'{evalFolder_name_base}_{iteration+iter_offset}'

        for model_path in models:
            for epoch in epochs:
                os.system(f'python {root}/tools/result_parser_katrin_longest6.py --results {root}/{model_path}/{evalFolder_name}/{evaluation_benchmark}/epoch{epoch}')