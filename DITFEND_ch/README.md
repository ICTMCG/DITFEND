# Run
Take the 9-domain experiment (Finance as the target domain) for an example: 
## Step 1: General Model Training
```
python main.py --meta_task 9 --finetune_domain -1  --model_name bert --emb_type bert --global_lr 0.0007
```
## Step 2: Transferability Quantifying
Select data of finance domain from the folder ``/data/'' to construct ``train_finance.csv'' and ``val_finance.csv'',
```
python run_mlm.py --model_name_or_path bert-base-chinese --train_file train_finance.csv --validation_file val_finance.csv --do_train True --do_eval True --output_dir /language_models/finetune/
```
## Step 3: Domain Adaptation
```
python main.py --meta_task 9 --finetune_domain 0 --target_domain -1 --model_name bert --emb_type bert --finetune_lr 0.0007
```
