#!/usr/bin/env bash
# Single-GPU / single-node torchrun environment.
# Run this from dicee/run_scripts before launching an experiment:
#   source env_single_gpu.sh

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
