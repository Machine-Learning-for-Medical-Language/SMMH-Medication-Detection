python train_system.py \
--model_name_or_path /home/dongfangxu/Projects/models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
--data_dir /home/dongfangxu/Projects/SSN_Drug_Normalization/data/bert/classifier/smm4h20+/ \
--output_dir /home/dongfangxu/Projects/SSN_Drug_Normalization/data/models/test1/ \
--task_name smm4h \
--do_train \
--do_eval \
--num_train_epochs 10 \
--train_batch_size 32 \
--overwrite_output_dir true \
--overwrite_cache true