# 18
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.001.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.0005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.0010.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.00005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/18/0.00001.yaml
sleep 60

# 34
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.001.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.0005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.0010.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.00005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/34/0.00001.yaml
sleep 60

# 50
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.001.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.0005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.0010.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.00005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/50/0.00001.yaml
sleep 60

# 101
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.001.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.0005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.0010.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.00005.yaml
sleep 60
torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg experiments/baseline/101/0.00001.yaml