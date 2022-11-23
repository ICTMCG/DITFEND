# Run
## Step 1: General Model Training
```
python main.py --meta_task 3 --finetune_domain -1 --target_domain -1 --model_name textcnn --emb_type w2v --global_lr 0.0009
```
## Step 2: Transferability Quantifying
```

```
## Step 3: Domain Adaptation
```
python main.py --meta_task 3 --finetune_domain 0 --target_domain -1 --model_name textcnn --emb_type w2v --finetune_lr 0.0009
```
