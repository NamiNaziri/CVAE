#!/bin/bash


#SBATCH --time=100:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-p100-16g,gpu-v100-32g,gpu-v100-16g



echo "$1"
experiment_name=$1
echo ""$experiment_name".pi"
checkpoint=$2
kld_weight=$3
latent_size=$4


echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

# For the next line to work, you need to be in the
# hpc-examples directory.
#srun python3 train_eval_combo.py --experiment="$experiment_name"

#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_cvae_light_ar_cond_dof2.py --run_name="triton" --out_name=$experiment_name  --kld_weight=$kld_weight --latent_size=$latent_size
#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_cvae_light_ar_cond_part_wise_dof.py --run_name="triton" --out_name=$experiment_name  --kld_weight=$kld_weight --latent_size=$latent_size
#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_cvae_light_ar_cond_dof_lowerOnly.py --run_name="triton" --out_name=$experiment_name  --kld_weight=$kld_weight --latent_size=$latent_size
#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_cvae_light_ar_cond_dof_upperOnly.py --run_name="triton" --out_name=$experiment_name  --kld_weight=$kld_weight --latent_size=$latent_size

#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_cvae_light_ar_cond_dof_rootOnly.py --run_name="triton" --out_name=$experiment_name  --kld_weight=$kld_weight --latent_size=$latent_size

#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_wvae_light_ar_cond_dof_rootOnly.py --run_name="triton" --out_name=$experiment_name  --lambda_val=$kld_weight --latent_size=$latent_size
#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_wvae_light_ar_cond_dof_lowerOnly.py --run_name="triton" --out_name=$experiment_name  --lambda_val=$kld_weight --latent_size=$latent_size
#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_root_humor.py --run_name="triton" --out_name=$experiment_name  --kld_weight=$kld_weight --latent_size=$latent_size
#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_root_humor2.py --run_name="triton" --out_name=$experiment_name  --kld_weight=$kld_weight --latent_size=$latent_size
#srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_wvae_light_ar_cond_dof_fullBody_3.py --run_name="triton" --out_name=$experiment_name  --lambda_val=$kld_weight --latent_size=$latent_size

srun singularity run --nv --bind /scratch /home/nazirin1/scratch/AGit/MCPHC/opengl.sif python triton/$experiment_name/train_cvae_light_ar_cond_dof3_triton.py --run_name="triton" --out_name=$experiment_name  --kld_weight=$kld_weight --latent_size=$latent_size

