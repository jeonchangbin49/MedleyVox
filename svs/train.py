import sys
import time

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import wandb


from .solver import Solver


def train(args):
    print("hello")
    solver = Solver()

    ngpus_per_node = int(torch.cuda.device_count() / args.n_nodes)
    print(f"use {ngpus_per_node} gpu machine")
    args.world_size = ngpus_per_node * args.n_nodes
    mp.spawn(worker, nprocs=ngpus_per_node, args=(solver, ngpus_per_node, args))


def worker(gpu, solver, ngpus_per_node, args):
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend="nccl", world_size=args.world_size, init_method="env://", rank=args.rank
    )
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    solver.set_gpu(args)

    start_epoch = solver.start_epoch

    for epoch in range(start_epoch, args.epochs + 1):

        solver.train(args, epoch)

        time.sleep(1)

        solver.multi_validate(args, epoch)

        if solver.stop == True:
            print("Apply Early Stopping")
            if args.use_wandb:
                wandb.finish()
            sys.exit()

    if args.use_wandb:
        wandb.finish()
