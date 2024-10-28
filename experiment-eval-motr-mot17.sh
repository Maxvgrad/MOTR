#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-internship-eval-motr-mot17"
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=not_tracked_dir/slurm/%j_slurm_%x.out

# Resource allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G

# Node configurations (commented out)
## falcone configurations
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4

## Pegasus configuration
##SBATCH --gres=gpu:a100-40g:4
##SBATCH --cpus-per-task=24

## Pegasus2 configuration
##SBATCH --gres=gpu:a100-80g:2
##SBATCH --cpus-per-task=16

#----------------------------------------
# Parse Arguments
#----------------------------------------
resume_flag=false
while getopts "r" opt; do
  case ${opt} in
    r )
      resume_flag=true
      ;;
    \? )
      echo "Usage: cmd [-r] to enable resume"
      exit 1
      ;;
  esac
done

#----------------------------------------
# Environment Setup
#----------------------------------------
module load miniconda3
conda activate motr

#----------------------------------------
# Directory Setup
#----------------------------------------
# Create output directory with timestamp
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
output_dir="not_tracked_dir/output_${model_name}_${timestamp}"
# For debugging/testing, use fixed output directory
output_dir="models/crowdhuman_deformable_multi_frame_reproduce"

EXP_DIR=exps/e2e_motr_r50_joint
MOT_ROOT_DIR=/gpfs/helios/home/ploter/projects/datasets

#----------------------------------------
# Distributed Training Setup
#----------------------------------------
# Set master node address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12381

# Calculate world size for distributed training
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

# Enable NCCL debugging
export NCCL_DEBUG=INFO

#----------------------------------------
# Debug Information
#----------------------------------------
echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "RANK=$RANK"
echo "output_dir=$output_dir"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

#----------------------------------------
# Training Command
#----------------------------------------

# Set parameters based on resume flag
if [ "$resume_flag" = true ]; then
    checkpoint=$output_dir/checkpoint.pth
    wandb_id="djiokxzk"
    resume_optim=true
    resume=$checkpoint
    echo "Resuming training from checkpoint: $checkpoint"
else
    wandb_id=None  # Set this to an empty string or a default value if necessary
    resume_optim=false
    resume=""  # No resume
    echo "Starting new training run."
fi

python3 main.py \
   --meta_arch motr \
   --use_checkpoint \
   --dataset_file e2e_joint \
   --epoch 200 \
   --with_box_refine \
   --lr_drop 100 \
   --lr 2e-4 \
   --lr_backbone 2e-5 \
   --pretrained ${EXP_DIR}/motr_final.pth \
   --output_dir ${EXP_DIR} \
   --batch_size 1 \
   --sample_mode 'random_interval' \
   --sample_interval 10 \
   --sampler_steps 50 90 120 \
   --sampler_lengths 2 3 4 5 \
   --update_query_pos \
   --merger_dropout 0 \
   --dropout 0 \
   --random_drop 0.1 \
   --fp_ratio 0.3 \
   --query_interaction_layer 'QIM' \
   --extra_track_attn \
   --data_txt_path_train ./datasets/data_path/joint.train \
   --data_txt_path_val ./datasets/data_path/mot17.train \
   --resume ${EXP_DIR}/motr_final.pth \
   --mot_path ${MOT_ROOT_DIR} \
   --eval