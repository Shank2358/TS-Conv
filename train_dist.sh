MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    trainv2.py
