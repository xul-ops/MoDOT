

# CUDA_VISIBLE_DEVICES=1 python3 modot/wandb_train.py configs/ssr/ob-future_train.txt

# CUDA_VISIBLE_DEVICES=1 python3 modot/eval_final.py configs/ssr/ob-future_eval.txt

# CUDA_VISIBLE_DEVICES=0 python3 modot/wandb_train.py configs/ssr/ob-hypersim_train.txt

# CUDA_VISIBLE_DEVICES=0 python3 modot/eval_final.py configs/ssr/ob-hypersim_eval.txt

CUDA_VISIBLE_DEVICES=0 python3 modot/wandb_train.py configs/ssr/nyudmt_train.txt

CUDA_VISIBLE_DEVICES=0 python3 modot/eval_final.py configs/ssr/nyudmt_eval.txt



