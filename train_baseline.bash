# 18
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.001.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.0005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.0010.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.00005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.00001.yaml

# 34
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.001.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.0005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.0010.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.00005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.00001.yaml

# 50
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.001.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.0005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.0010.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.00005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.00001.yaml

# 101
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.001.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.0005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.0010.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.00005.yaml
# torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.00001.yaml

# without bias of regressor
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50_wo_fc_bias/0.001.yaml
# 1-layer regressor
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/wo_fc/0.001.yaml