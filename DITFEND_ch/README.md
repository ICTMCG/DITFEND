# Run
## Step 1: General Model Training
```
python main.py --meta_task 3 --finetune_domain -1  --model_name bert --emb_type bert --global_lr 0.0007
```
## Step 2: Transferability Quantifying
```
python run_mlm.py --model_name_or_path bert-base-chinese --train_file train_finance.csv --validation_file val_finance.csv --do_train True --do_eval True --output_dir /language_models/finetune/
```
## Step 3: Domain Adaptation
```
python main.py --meta_task 3 --finetune_domain 0 --target_domain -1 --model_name textcnn --emb_type w2v --finetune_lr 0.0009
```
