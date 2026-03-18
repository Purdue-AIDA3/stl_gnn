# Data generation
# python -m scripts  --out results/train_ex1.jsonl.gz  --example_id 1  --n 3000  --seed 0  --n_obstacles 3
# python -m scripts  --out results/train_ex2.jsonl.gz  --example_id 2  --n 6000  --seed 0  --n_obstacles 3
# python -m scripts  --out results/train_ex3.jsonl.gz  --example_id 3  --n 3000  --seed 0  --n_obstacles 4
# python -m scripts  --out results/train_ex4.jsonl.gz  --example_id 4  --n 6000  --seed 0  --n_obstacles 3
# python -m scripts  --out results/train_ex5.jsonl.gz  --example_id 5  --n 3000  --seed 0  --n_obstacles 5


# GNN: training
# python -m gnn_train.train  --data results/train_ex1.jsonl.gz  --example_id 1  --epochs 500  --summary_out results/summary_ex1.png  --ckpt_out results/ckpt_ex1.pt   
# python -m gnn_train.train  --data results/train_ex2.jsonl.gz  --example_id 2  --epochs 500  --summary_out results/summary_ex2.png  --ckpt_out results/ckpt_ex2.pt
# python -m gnn_train.train  --data results/train_ex3.jsonl.gz  --example_id 3  --epochs 500  --summary_out results/summary_ex3.png  --ckpt_out results/ckpt_ex3.pt
# python -m gnn_train.train  --data results/train_ex4.jsonl.gz  --example_id 4  --epochs 500  --summary_out results/summary_ex4.png  --ckpt_out results/ckpt_ex4.pt
# python -m gnn_train.train  --data results/train_ex5.jsonl.gz  --example_id 5  --epochs 500  --summary_out results/summary_ex5.png  --ckpt_out results/ckpt_ex5.pt

# CNN: training
# python -m cnn_train.cnn_train  --data results/train_ex1.jsonl.gz  --example_id 1  --epochs 500  --ckpt_out results/cnn_ckpt_ex1.pt 
# python -m cnn_train.cnn_train  --data results/train_ex2.jsonl.gz  --example_id 2  --epochs 500  --ckpt_out results/cnn_ckpt_ex2.pt
# python -m cnn_train.cnn_train  --data results/train_ex3.jsonl.gz  --example_id 3  --epochs 500  --ckpt_out results/cnn_ckpt_ex3.pt
# python -m cnn_train.cnn_train  --data results/train_ex4.jsonl.gz  --example_id 4  --epochs 500  --ckpt_out results/cnn_ckpt_ex4.pt
# python -m cnn_train.cnn_train  --data results/train_ex5.jsonl.gz  --example_id 5  --epochs 500  --ckpt_out results/cnn_ckpt_ex5.pt

# MLP: training
# python -m mlp_train.mlp_train  --data results/train_ex1.jsonl.gz  --example_id 1  --epochs 500  --ckpt_out results/mlp_ckpt_ex1.pt
# python -m mlp_train.mlp_train  --data results/train_ex2.jsonl.gz  --example_id 2  --epochs 500  --ckpt_out results/mlp_ckpt_ex2.pt
# python -m mlp_train.mlp_train  --data results/train_ex3.jsonl.gz  --example_id 3  --epochs 500  --ckpt_out results/mlp_ckpt_ex3.pt
# python -m mlp_train.mlp_train  --data results/train_ex4.jsonl.gz  --example_id 4  --epochs 500  --ckpt_out results/mlp_ckpt_ex4.pt
# python -m mlp_train.mlp_train  --data results/train_ex5.jsonl.gz  --example_id 5  --epochs 500  --ckpt_out results/mlp_ckpt_ex5.pt

# GNN: evaluation
python -m gnn_train.eval_qp  --ckpt results/ckpt_ex1.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/eval_ex1  --example_id 1   --n_obstacles 3   --topk 1000
python -m gnn_train.eval_qp  --ckpt results/ckpt_ex2.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/eval_ex2  --example_id 2   --n_obstacles 3   --topk 1000
python -m gnn_train.eval_qp  --ckpt results/ckpt_ex3.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/eval_ex3  --example_id 3   --n_obstacles 4   --topk 1000
python -m gnn_train.eval_qp  --ckpt results/ckpt_ex4.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/eval_ex4  --example_id 4   --n_obstacles 3   --topk 1000
python -m gnn_train.eval_qp  --ckpt results/ckpt_ex5.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/eval_ex5  --example_id 5   --n_obstacles 5   --topk 1000

# CNN: evaluation
# python -m cnn_train.cnn_eval_qp  --ckpt results/cnn_ckpt_ex1.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/cnn_eval_ex1  --example_id 1  --n_obstacles 3  --topk 1000
# python -m cnn_train.cnn_eval_qp  --ckpt results/cnn_ckpt_ex2.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/cnn_eval_ex2  --example_id 2  --n_obstacles 3  --topk 1000
# python -m cnn_train.cnn_eval_qp  --ckpt results/cnn_ckpt_ex3.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/cnn_eval_ex3  --example_id 3  --n_obstacles 4  --topk 1000
# python -m cnn_train.cnn_eval_qp  --ckpt results/cnn_ckpt_ex4.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/cnn_eval_ex4  --example_id 4  --n_obstacles 3  --topk 1000
# python -m cnn_train.cnn_eval_qp  --ckpt results/cnn_ckpt_ex5.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/cnn_eval_ex5  --example_id 5  --n_obstacles 5  --topk 1000

# MLP: evaluation
# python -m mlp_train.mlp_eval_qp  --ckpt results/mlp_ckpt_ex1.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/mlp_eval_ex1  --example_id 1  --n_obstacles 3  --topk 1000
# python -m mlp_train.mlp_eval_qp  --ckpt results/mlp_ckpt_ex2.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/mlp_eval_ex2  --example_id 2  --n_obstacles 3  --topk 1000
# python -m mlp_train.mlp_eval_qp  --ckpt results/mlp_ckpt_ex3.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/mlp_eval_ex3  --example_id 3  --n_obstacles 4  --topk 1000
# python -m mlp_train.mlp_eval_qp  --ckpt results/mlp_ckpt_ex4.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/mlp_eval_ex4  --example_id 4  --n_obstacles 3  --topk 1000
# python -m mlp_train.mlp_eval_qp  --ckpt results/mlp_ckpt_ex5.pt  --n 100  --run_micp_baseline 0  --save_png 1  --png_dir results/mlp_eval_ex5  --example_id 5  --n_obstacles 5  --topk 1000

# STLPY: evaluation (MICP, no training required)
# python -m stlpy_eval.eval_micp  --n 100  --seed 1  --example_id 1  --n_obstacles 3  --save_png 1  --png_dir results/stlpy_eval_ex1
# python -m stlpy_eval.eval_micp  --n 100  --seed 1  --example_id 2  --n_obstacles 3  --save_png 1  --png_dir results/stlpy_eval_ex2
# python -m stlpy_eval.eval_micp  --n 100  --seed 1  --example_id 3  --n_obstacles 4  --save_png 1  --png_dir results/stlpy_eval_ex3
# python -m stlpy_eval.eval_micp  --n 100  --seed 1  --example_id 4  --n_obstacles 3  --save_png 1  --png_dir results/stlpy_eval_ex4
# python -m stlpy_eval.eval_micp  --n 100  --seed 1  --example_id 5  --n_obstacles 5  --save_png 1  --png_dir results/stlpy_eval_ex5