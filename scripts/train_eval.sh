


# CUDA_VISIBLE_DEVICES=0 python3 modot/wandb_train.py configs/modot/ob-future_train.txt

# CUDA_VISIBLE_DEVICES=0 python3 modot/eval_final.py configs/modot/ob-future_eval.txt

# CUDA_VISIBLE_DEVICES=0 python3 modot/wandb_train.py configs/modot/ob-hypersim_train.txt

# CUDA_VISIBLE_DEVICES=0 python3 modot/eval_final.py configs/modot/ob-hypersim_eval.txt

CUDA_VISIBLE_DEVICES=1 python3 modot/wandb_train.py configs/modot/nyudmt_train.txt

CUDA_VISIBLE_DEVICES=1 python3 modot/eval_final.py configs/modot/nyudmt_eval.txt



