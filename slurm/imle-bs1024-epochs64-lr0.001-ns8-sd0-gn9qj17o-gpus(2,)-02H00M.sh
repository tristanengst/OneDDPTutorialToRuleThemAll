#!/bin/bash
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --exclude=
#SBATCH --constraint=

# I/O and email notification setup ###################################################
#SBATCH --job-name=preempt_me_imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o-gpus(2,)-02H00M
#SBATCH --output=pretrain_results/imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o-gpus(2,)-02H00M.txt
#SBATCH --open-mode=append

export OMP_NUM_THREADS=1

########### Contextual info for job ##################################################
printf "\n\n"
echo "[`date`] Starting job=$SLURM_JOB_ID"
srun  -N $SLURM_NNODES -n $SLURM_NNODES bash -c ' echo "[`date`] Working directory=`pwd`    node=$(hostname)" '
export SLURM_TMPDIR=/localscratch/$USER/SLURM_TMPDIRS/imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o

srun  -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF 
mkdir -p $SLURM_TMPDIR ; rm -rf $SLURM_TMPDIR/*
EOF
wait

# Basic Python installation assumed to exist already

srun  -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF 
ln -s /project/apex-lab/$USER/scratch ~/scratch
EOF
wait


srun  -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF 
mkdir -p /localscratch/$USER/data ; ln -s /localscratch/$USER/data $SLURM_TMPDIR/data
EOF
wait



export PYTHONUNBUFFERED=1
export MASTER_PORT=41278
export MASTER_ADDR=$(hostname)
export TRITON_HOME=$SLURM_TMPDIR
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=1
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

echo "r$SLURM_NODEID master: $MASTER_ADDR"

############ Move code to model folder if first task #################################
need_to_tar_code=$(python UtilsSlurm.py --before_training --function need_to_tar_code --exp MODEL_FOLDER)
echo "[`date`] need_to_tar_code=$need_to_tar_code"
if $need_to_tar_code; then
    mkdir -p /home/tme3/Development/DDPExample/runs/imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o
	cur_git_sha=$(python UtilsSlurm.py --function get_git_sha)
	echo $cur_git_sha > cur_git_sha.txt
	tar -rf /home/tme3/Development/DDPExample/runs/imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o/code.tar *.py
	tar -rf /home/tme3/Development/DDPExample/runs/imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o/code.tar cur_git_sha.txt
else
    echo "[`date`] Code already tarred"
fi

############ Detect whether the current task should run, set persisted date ##########
python UtilsSlurm.py --before_training --function startup_check --exp MODEL_FOLDER --submit_dir /home/tme3/Development/DDPExample --slurm_script /home/tme3/Development/DDPExample/slurm/imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o-gpus(2,)-02H00M.sh

############ Move code to the compute node ###########################################

srun  -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF 
tar -xf /home/tme3/Development/DDPExample/runs/imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o/code.tar -C $SLURM_TMPDIR
EOF
wait


srun -N $SLURM_NNODES -n $SLURM_NNODES nvidia-smi

############ Training. Use srun --time to ensure time limit is enforced ##############

srun  -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF 
python UtilsSlurm.py --function copy --cp_src MNIST --cp_parent $SLURM_TMPDIR/ --cp_force 0 --untar_if_tar 0
EOF
wait


cd $SLURM_TMPDIR

python UtilsSlurm.py --function copy --cp_src /home/tme3/Development/DDPExample/py311OneDDPTutorial.tar.gz --cp_parent $SLURM_TMPDIR --untar_if_tar 0 --cp_force -1
mkdir -p $SLURM_TMPDIR/py311OneDDPTutorial
tar -xzf $SLURM_TMPDIR/py311OneDDPTutorial.tar.gz -C $SLURM_TMPDIR/py311OneDDPTutorial
$SLURM_TMPDIR/py311OneDDPTutorial/bin/python
source $SLURM_TMPDIR/py311OneDDPTutorial/bin/activate



timelimit=$(python UtilsSlurm.py --function get_time_limit --epi_time 2)
srun --nodes 1 --ntasks-per-node 2 --cpus-per-task 8 --time $timelimit bash -c ' python TrainAndEval.py  --gpus 0 1 --resume latest --save_iter_t 7 --speedup ddp --uid gn9qj17o --wandb online   --job_id $SLURM_JOB_ID --num_workers 8 --compute_node $SLURM_TMPDIR --save_folder ~/scratch/IMLE-SSL --submit_dir /home/tme3/Development/DDPExample'

############ Detect and implement finishing conditions if time permits ###############
python UtilsSlurm.py --function shutdown_check --exp MODEL_FOLDER --slurm_script /home/tme3/Development/DDPExample/slurm/imle-bs1024-epochs64-lr0.001-ns8-sd0-gn9qj17o-gpus(2,)-02H00M.sh --epi_time 2  --submit_dir /home/tme3/Development/DDPExample

########### Contextual info for job ##################################################
echo "[`date`] Ending job=$SLURM_JOB_ID"
echo "[`date`] Working directory=`pwd`    node=`hostname`"

