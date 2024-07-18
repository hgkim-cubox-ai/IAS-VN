# ResNet-34
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/34_0.0005.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/34_0.0001.yaml
torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/34/0.00005.yaml

# ResNet-50
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/50/0.01.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/50/0.005.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/50/0.001.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/50/0.0005.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/50/0.0001.yaml
torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/50/0.00005.yaml

# ResNet-101
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/101/0.01.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/101/0.005.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/101/0.001.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/101/0.0005.yaml
# torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/101/0.0001.yaml
torchrun --nnodes=1 --nproc_per_node=4 main.py --cfg experiments/baseline/101/0.00005.yaml