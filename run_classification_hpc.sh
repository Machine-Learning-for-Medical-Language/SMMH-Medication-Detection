#!/bin/bash
#SBATCH --partition=chip-gpu             # queue to be used
#SBATCH --account=chip
#SBATCH --time=24:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --job-name=conorm             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL      # send and email when the job begins, ends or fails
#SBATCH --mail-user=dongfang.xu@childrens.harvard.edu      # Email address to send the job status
#SBATCH --output=log/rand_0.45_0.05uniformity_0.1dropout%j.txt    # Name of the output file
#SBATCH --error=log/rand_0.45_0.05uniformity_0.1dropout%j.err
#SBATCH --nodes=1               # Number of gpu nodes
#SBATCH --gres=gpu:Titan_RTX:1                # Number of gpu devices on one gpu node


pwd; hostname; date

module load singularity

singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif  python3.7 train_system.py \
--model_name_or_path /temp_work/ch223150/outputs/model/Bio_ClinicalBERT \
--data_dir /temp_work/ch223150/outputs/smm4h/data/classification/smm4h20+_nertoclassifer_upsampling/ \
--output_dir /temp_work/ch223150/outputs/smm4h/models/classification/upsampling_biobert/ \
--task_name smm4h \
--do_train \
--do_eval \
--learning_rate 3e-5 \
--per_device_train_batch_size 32 \
--max_seq_length 128 \
--num_train_epochs 10 \
--overwrite_output_dir true \
--overwrite_cache true