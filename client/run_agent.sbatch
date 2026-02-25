#!/bin/batch_size

#SBATCH --job-name=RunAgent
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=3G
#SBATCH --time=13:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kai.luedemann@student.uni-tuebingen.de

cp -R ~/rl-project /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/rl-project

singularity exec /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg python -m venv --system-site-packages venv
singularity exec /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg ./venv/bin/pip install -r requirements.txt
singularity exec /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg ./script2.sh &
singularity exec /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg ./script2.sh &
singularity exec /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg ./script2.sh &
singularity exec /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg ./script2.sh &
singularity exec /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg ./script2.sh
