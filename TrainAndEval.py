import argparse
import copy
import einops
import math
import numpy as np
import os
import os.path as osp
import random
import psutil
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms.v2 as v2
import wandb


import Utils
from Utils import twrite, tqdm_wrap
import UtilsSlurm

######################################################################################
# NOT INTERESTING FOR DDP, SKIP TO THE NEXT SECTION (around line 200)
######################################################################################
class LinearRampThenCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    """Linear ramp from zero to [args.lr] over [args.warmup_epochs] epochs, then
    cosine decay to zero over the remaining training epochs.

    Args:
    optimizer           -- the optimizer to wrap
    lrs                 -- the learning rates to use
    last_epoch          -- index of the last gradient step (-1 at intialization)
    args                -- argparse Namespace containing the arguments
    steps_per_epoch     -- the number of steps per epoch
    """
    def __init__(self, optimizer, *, last_epoch=-1, args, steps_per_epoch):
        self.args = args
        self.last_epoch = last_epoch        
        warmup = torch.linspace(0, args.lr, steps_per_epoch * args.warmup_epochs)
        total_cosine_epochs = (args.epochs - args.warmup_epochs) * steps_per_epoch
        cosine = torch.tensor([args.lr * (1 + math.cos(math.pi * t / (total_cosine_epochs))) / 2 for t in range(total_cosine_epochs)])
        self.step2lr = torch.cat([warmup, cosine])
        super(LinearRampThenCosineDecay, self).__init__(optimizer)
        
    def get_lr(self): return self.step2lr[self.last_epoch]
    def get_last_lr(self): return self.step2lr[self.last_epoch]
    def step(self, epoch=None):
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr()
        self.last_epoch = epoch

def get_experiment_name(args, make_folder=False):
    """Returns the unique name of an experiment."""
    return f"imle-bs{args.bs}-epochs{args.epochs}-lr{args.lr}-ns{args.ns}-sd{args.seed}-{args.uid}"

def get_save_folder(args, make_folder=False):
    """Returns the folder to save checkpoints and logs in.

    Args:
    args        -- argparse Namespace containing the arguments
    make_folder -- whether to create the folder if it doesn't exist
    """
    folder = f"{osp.dirname(__file__)}/runs/{get_experiment_name(args)}"
    _ = os.makedirs(folder, exist_ok=True) if make_folder else None
    return folder

# Just used for data augmentation
def min_max_normalization(tensor, min_value=0, max_value=1):
    min_tensor, max_tensor = torch.min(tensor), torch.max(tensor)
    tensor = (tensor - min_tensor)
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

# A neural network that corrupts input images and tries to reconstruct them. Wrapped
# with an IMLEWrapper, we can do cIMLE training.
class MaskedAutoencoder(nn.Module):
    """Denoising IMLE autoencoder with a random mask applied to the input."""
    def __init__(self, args):
        super(MaskedAutoencoder, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(nn.Flatten(),
            nn.Linear(args.img_size ** 2, args.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.h_dim, args.latent_dim))
        self.decoder = nn.Sequential(nn.Linear(args.latent_dim, args.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.h_dim, args.img_size ** 2))

    @torch.compiler.disable
    def get_masks(self, bs, *, device, mask_z=None):
        """Returns [bs] random binary masks for the input images on device [device]
        with optional seed [mask_z].
        """
        g = None if mask_z is None else torch.Generator("cpu").manual_seed(mask_z)
        pps = self.args.img_size // self.args.patch_size
        mask = torch.rand(bs, pps * pps, device=device, generator=g)
        mask = torch.where(mask < self.args.mask_ratio, 0, 1)
        mask = einops.rearrange(mask, "bs (pps1 pps2) -> bs pps1 pps2",
            pps1=pps, pps2=pps)
        mask = einops.repeat(mask, "bs pps1 pps2 -> bs (pps1 h) (pps2 w)",
            h=self.args.patch_size, w=self.args.patch_size, pps1=pps, pps2=pps)
        return mask

    @torch.compiler.disable
    def get_codes(self, nz, *, device, z=None):
        """Returns [nz] random codes on device [device] with optional seed [z]."""
        g = None if z is None else torch.Generator("cpu").manual_seed(z)
        return torch.randn(nz, self.args.latent_dim, device=device, generator=g)

    @torch.no_grad()
    def random_masking(self, x, mask_z=None):
        """Returns a randomly-masked version of tensor [x]. [mask_z] should be either
        an appropriately-sized mask tensor, 

        Args:
        x      -- NxHxW tensor to mask
        mask_z -- either an NxHxW mask tensor, a seed to sample the tensor with, or
                    None to use the current state of randomness
        """
        if mask_z is None:
            mask_z = self.get_masks(x.shape[0], device=x.device, mask_z=mask_z) 
        return x * einops.rearrange(mask_z, "n h w -> n 1 h w")

    def forward_img(self, x, mask_z=None, z=None, ns=None):
        """Returns the masked and predicted images for input [x] with optional mask
        [mask_z] and latent codes [z].

        Args:
        x       -- NxCxHxW input tensor
        mask_z  -- For random masking. Either an NxHxW mask tensor, a seed to
                    sample with, or None to use the current state of randomness
        z       -- For latent codes. Either an NxHxW mask tensor, a seed to sample
                    with, or None to use the current state of randomness
        ns      -- number of latent codes to sample if [z] isn't a tensor. If None,
                    uses [args.ns]
        """
        bs, c, h, w = x.shape
        if not isinstance(z, torch.Tensor):
            ns = args.ns if ns is None else ns
            z = self.get_codes(x.shape[0] * ns, device=x.device, z=z)
 
        x_masked = self.random_masking(x, mask_z=mask_z)
        fx = self.encoder(x_masked)
        fx = einops.repeat(fx, "n d -> (s n) d", s=z.shape[0] // x.shape[0])
        fx = fx * (1 + torch.nn.functional.relu(z))
        fx = self.decoder(fx)
        fx = einops.rearrange(fx, "n (c h w) -> n c h w", h=h, w=w, c=c)
        return x_masked, fx
    
    def forward(self, x, mask_z=None, z=None, reduction="mean"):
        """Returns the loss from trying to reconstruct [x].
        
        Args:
        x           -- NxCxHxW input tensor
        reduction   -- the reduction method for the loss ('batch', 'sum' or 'mean')
        """
        _, fx = self.forward_img(x, mask_z=mask_z, z=z)
        y = einops.repeat(x, "n c h w -> (s n) c h w", s=fx.shape[0] // x.shape[0])
        loss = nn.functional.mse_loss(fx, y, reduction="none")
        loss = einops.reduce(loss, "n c h w -> n", "mean")
        return loss if reduction == "batch" else einops.reduce(loss, "n -> ()", reduction)

# Wrapper class that allows cIMLE training. A call to the forward() method computes
# the loss with min-loss latent codes. The update_sampler() method updates the weights
# used to figure out which latent code is best to match those being trained. Latent
# codes are never saved and it's impossible to train on one multiple times.
class IMLEWrapper(nn.Module):
    """Wrapper for a model to enable cIMLE.
    
    Args:
    args    -- argparse Namespace containing the arguments
    model   -- the model to wrap
    """
    def __init__(self, *, args, model):
        super(IMLEWrapper, self).__init__()
        self.args = args
        self.model = model
        self.sampler = copy.deepcopy(model)
        self.update_sampler()

    def update_sampler(self):
        """Updates the parameters of the sampler to match those of the model."""
        for p,sp in zip(self.model.parameters(), self.sampler.parameters()):
            sp.data = p.data.clone()
        self.sampler.requires_grad_(False)

    def forward(self, x, mask_z=None, z=None, reduction="mean"):
        """Returns the loss from trying to reconstruct [x].
        
        Args:
        x           -- Nx... input tensor
        mask_z      -- either an NxHxW mask tensor, a seed to sample with, or None to
                        use the current state of randomness
        z           -- either an NxHxW mask tensor, a seed to sample with, or None to
                        use the current state of randomness
        reduction   -- the reduction method for the loss
        """
        bs, _, _, _ = x.shape
        if not isinstance(mask_z, torch.Tensor):
            mask_z = self.sampler.get_masks(bs, device=x.device, mask_z=mask_z)
        if not isinstance(z, torch.Tensor):
            z = self.sampler.get_codes(bs * args.ns, device=x.device, z=z)
        
        with torch.no_grad():
            loss = self.sampler(x, mask_z=mask_z, z=z, reduction="batch")
            min_loss_idxs = torch.argmin(loss.view(bs, args.ns), dim=1)
            z = z.view(bs, args.ns, -1)[torch.arange(bs, device=z.device), min_loss_idxs]

        # Something very important here! We are returning the average loss across
        # whatever the batch-size-per-GPU samples is with reduction='mean'
        return self.model(x, mask_z=mask_z, z=z, reduction=reduction)


######################################################################################
######################################################################################
######################################################################################




######################################################################################
# BEGINS TO BE INTERESTING FOR DDP
######################################################################################

# Why we have a special function to get argparse arguments will be come apparent when
# we look at SLURMification.
def get_args(args=None, on_compute_node=True):
    """Returns an argparse Namespace for training.

    Args:
    args            -- unparsed argparse arguments
    on_compute_node -- if being run on the hardware where training will be performed
    """
    P = argparse.ArgumentParser()
    
    # Specifies what hardware to use and how to use it
    P.add_argument("--gpus", type=int, default=[0], nargs="+",
        help="GPU indices to use")
    P.add_argument("--nodes", type=int, default=1,
        help="Number of compute nodes")
    P.add_argument("--speedup", default="compile_ddp", choices=["gpu", "compile", "dataparallel", "ddp", "compile_ddp"],
        help="Speedup method")
    P.add_argument("--hw_mp_ctx", default="adapt", choices=["fork", "spawn", "adapt"],
        help="Hardware multiprocessing context")
    P.add_argument("--hw_dist_backend", default="nccl", choices=["nccl", "gloo"],
        help="Hardware distributed backend")
    P.add_argument("--num_workers", type=int, default=8,
        help="Number of workers for loading data")
    P.add_argument("--autocast", type=bool, default=True, choices=[0, 1],
        help="Whether to use automatic mixed precision")

    # Training hyperparameters and related settings
    P.add_argument("--bs", type=int, default=1024,
        help="(effective) batch size")
    P.add_argument("--accum_iter", type=int, default=1,
        help="number of iterations to accumulate gradients over")
    P.add_argument("--lr", type=float, default=1e-3,
        help="Maximum learning rate. We follow a linear-ramp-then-cosine-decay schedule")
    P.add_argument("--epochs", type=int, default=64,
        help="Number of epochs")
    P.add_argument("--warmup_epochs", type=int, default=5,
        help="Number of warmup epochs during which the learning rate ramps up")
    P.add_argument("--patch_size", type=int, default=4, 
        help="Size of patches to use in random masking")
    P.add_argument("--img_size", type=int, default=28, 
        help="Spatial size of input images")
    P.add_argument("--mask_ratio", type=float, default=0.5,
        help="Average ratio of pixels to mask out")
    P.add_argument("--latent_dim", type=int, default=8,
        help="Dimension of the latent space")
    P.add_argument("--h_dim", type=int, default=64,
        help="Dimension of hidden layers")

    # Training hyperparameters for cIMLE
    P.add_argument("--ns", type=int, default=8, 
        help="Number of latent codes to use in sampling")
    P.add_argument("--sampler_update_iter", type=int, default=1,
        help="Update the sampler every [sampler_update_iter] epochs")

    # Saving, loading, and run uniqueness
    P.add_argument("--eval_iter", type=int, default=4,
        help="Number of epochs between evaluations")
    P.add_argument("--save_iter", type=int, default=0,
        help="Number of epochs between saving checkpoints")
    P.add_argument("--save_iter_t", type=int, default=15,
        help="Number of minutes between saving '..._latest.pt' checkpoints")
    P.add_argument("--seed", type=int, default=0,
        help="Random seed")
    P.add_argument("--uid", default=None, 
        help="Unique identifier for the run. Set automatically if None.")
    P.add_argument("--resume", default="latest",
        help="Path to a checkpoint to resume from, 'none' to start from scratch, or 'latest' to resume from the latest checkpoint if it exists and from scratch otherwise")

    args = P.parse_args(args=args)
    
    # All processes will initially generate different UIDs. We can fix this after
    # initializing DDP.
    args.uid = wandb.util.generate_id() if args.uid is None else args.uid

    # Adaptively set --accum_iter, --bs_per_pass, and --bs_per_gpu based on the
    # current hardware. This means that you can just think in terms of numbers of GPUs
    # and everything else will work out!
    if on_compute_node:
        args = args_with_batch_size_args(args)

    return args

# Adaptively set batch size and related arguments based on the current hardware. This
# means that you can just think in terms of numbers of total GPUs and everything else
# will work out!
def args_with_batch_size_args(args):
    """Returns [args] with --accum_iter, --bs_per_pass, and --bs_per_gpu set."""
    def get_vram_per_gpu():
        """Returns the amount of VRAM per GPU in GB."""
        return min([int(torch.cuda.get_device_properties(0).total_memory * 9.313183594495e-10)
            for i in range(torch.cuda.device_count())])

    """Returns [args] with batch size-related arguments set."""
    # It's possible to have enough VRAM to ask for a larger batch size than there's
    # RAM to handle (cough cough A99). Use a heuristic to determine the result.
    def get_max_bs():
        """Returns the maximum possible batch size as a function of available VRAM."""
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        return float("inf")

    # Use a heuristic (eg. model size, image size) to determine the maximum batch size
    def get_max_bs_per_gpu(args):
        """Returns the heuristic maximum batch size that be processed on one GPU."""
        return 1024

    max_bs = get_max_bs()
    max_bs_per_gpu = get_max_bs_per_gpu(args)

    if args.accum_iter is None:
        for accum_iter in range(1, args.bs):
            if args.bs % accum_iter == 0:
                bs_per_pass = args.bs // accum_iter
                if bs_per_pass % len(args.gpus) == 0 and bs_per_pass <= max_bs:
                    bs_per_gpu = bs_per_pass // len(args.gpus)
                    if bs_per_gpu <= max_bs_per_gpu:
                        break
        raise ValueError("No valid batch size found")
    else:
        accum_iter = args.accum_iter
        bs_per_pass = args.bs // args.accum_iter
        bs_per_gpu = bs_per_pass // len(args.gpus)

    return argparse.Namespace(**vars(args) | dict(accum_iter=accum_iter,
        bs_per_pass=bs_per_pass, bs_per_gpu=bs_per_gpu))

# This function is roughly equivalent to constructing a normal DataLoader, but does so
# in a smart, adaptive way. The keyword arguments override corresponding ones in
# [args] if they are not None.
def get_loader(dataset, *, args, is_ddp=True, seed=None, batch_size=None, shuffle=False,
    num_workers=1, persistent_workers=False, multiprocessing_context="adapt", **kwargs):
    """Returns a (DistributedSampler, DataLoader) pair for training.

    Stochasticity for data order is controlled via [seed]. Stochasticity for data
    augmentation should be controlled through deterministically setting different
    seeds for each process.

    Args:
    args              -- argparse Namespace
    dataset           -- dataset to get dataloader for
    is_ddp            -- actually use DDP if args.distributed is True as well
    seed              -- seed for data order (not data augmentation). All processes
                        must set the same one, and should use the same one throughout
                        training. If None, uses [args.seed].
    batch_size        -- if [is_ddp] is True and in distributed mode, the per-GPU
                        batch size, otherwise the per-pass batch size. If None, uses
                        either of these arguments as found in [args]
    shuffle           -- whether to shuffle the data order
    num_workers       -- number of workers to use for loading data (per-process)
    persistent_workers -- whether to keep workers alive between epochs
    multiprocessing_context -- multiprocessing context to use if not given in [args].
                                One of 'fork', 'spawn', 'forkserver', or 'adapt'. See
                                the note below
    **kwargs          -- additional keyword arguments to pass to DataLoader
    """
    sampler = DistributedSampler(dataset,
        num_replicas=1 if not is_ddp else args.world_size,
        rank=Utils.get_rank(),
        seed=args.seed if seed is None else seed,
        shuffle=shuffle)
    
    # The multiprocessing context is crucial to get right. If not using DDP, it's safe
    # to leave it as None, which I believe is equivalent to 'fork'. However, with DDP,
    # 'fork' is unsafe for more than ~30 minutes. Both 'spawn' and 'forkserver' are
    # safe, with 'spawn' giving anecdotally slightly faster training. Both can take a
    # while to start.
    #
    # ---------- IMPORTANT NOTE WITH SPAWN -------------------------------------------
    # When DataLoader workers are spawned, they will run all code in this script not
    # gaurded by 'if __name__ == "__main__":'. This means that ungaurded code can slow
    # down their startup, and changing code during training can cause errors or
    # unexpected behavior.
    mp_ctx = args.hw_mp_ctx if multiprocessing_context is None else multiprocessing_context

    if is_ddp and args.distributed:
        mp_ctx = "spawn" if mp_ctx == "adapt" else mp_ctx
        batch_size = args.bs_per_gpu if batch_size is None else batch_size
    else:
        mp_ctx = "fork" if mp_ctx == "adapt" else mp_ctx
        batch_size = args.bs_per_pass if batch_size is None else batch_size

    # Observe that the DataLoader is constructed with the per-GPU batch size, not the
    # batch size that we actually want to use.
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
        num_workers=args.num_workers if num_workers is None else num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        multiprocessing_context=mp_ctx, **kwargs)

    return sampler, loader

# Ideally, whether we evaluate or not on a particular epoch shouldn't change the
# results of training. This decorator ensures nothing that happens in this function
# changes the state of randomness outside of it
@Utils.stateless
@torch.no_grad()
def evaluate(*, results, model, loader, args, num_images_to_visalize=6, num_samples_to_generate=4):
    """Returns [results] after adding the validation loss and other metrics.

    Args:
    results -- argparse Namespace containing the results to add to
    model   -- the model to evaluate
    loader  -- the validation DataLoader
    args    -- argparse Namespace containing experiment arguments
    """
    device = Utils.get_device(args)
    model.eval()

    # COMPUTE VALIDATION LOSS. There's some interesting concurrency to handle. Each
    # process will computed its own [loss_val] and [total]. We want them to
    # all agree on values for them. To this end, we use wrappers for dist.all_reduce()
    # on each.
    #
    # Note that this operation is SYNCHRONOUS, meaning that each process must have
    # computed a value first. 
    loss_val, total = 0, 0
    with torch.autocast("cuda", enabled=args.autocast):
        with Utils.no_sync_context(model):
            for x,_ in tqdm_wrap(loader, desc="Validation", leave=False):
                loss_val += model(x.to(device, non_blocking=True), reduction="sum")
                total += len(x)

    loss_val = Utils.all_reduce_sum(loss_val)
    total = Utils.all_reduce_sum(total)
    results.loss_val = (loss_val / total).item()

    # GENERATE SAMPLES. It's unlikely we want to generate many samples, so it's easier
    # to do this with just the main process. This means that [results] will be
    # different across processes for this field, but since the main process generates
    # images and the main process will log them, there's no issue.
    if Utils.dist_barrier_then_main():
        # Draw random images from the dataset
        idxs = torch.randint(0, len(loader.dataset), (8,))
        x = torch.stack([loader.dataset[idx][0] for idx in idxs], dim=0).to(device, non_blocking=True)
        x_masked, preds = Utils.as_plain_nn(model).model.forward_img(x, ns=num_samples_to_generate)
        
        # Not interesting for DDP!
        preds = einops.rearrange(preds, "(s n) c h w -> n s c h w", s=num_samples_to_generate)
        v = torch.cat([x.unsqueeze(1), x_masked.unsqueeze(1), preds], dim=1)
        v = einops.rearrange(v, "n s c h w -> (n s) c h w")
        images = torchvision.utils.make_grid(v, nrow=num_images_to_visalize, pad_value=1)
        _ = torchvision.utils.save_image(images, osp.join(get_save_folder(args, make_folder=True), f"epoch_{results.epoch}_samples.png"))
        _ = torchvision.utils.save_image(images, f"samples_{epoch}_prime.png")
        results.images = images
    else:
        results.images = None
    Utils.dist_barrier()
    return results

# By guarding most of the code with if __name__ == "__main__", we can run this script,
# spawned processes don't run it prior to being useful.
if __name__ == "__main__":
    ##################################################################################
    # Setup: (1) Get arguments, (2) Initialize DDP if being used, and get the device
    # of the current process, (3) All processes set the UID of the main process, (4)
    # All processes set the correct seed
    ##################################################################################
    args = get_args()
    args = Utils.args_with_ddp_args_set(args)
    args = Utils.init_distributed_mode(args)
    device = Utils.get_device(args)
    args.uid = Utils.from_main_process(args.uid)
    
    # The nth process sets the seed [args.seed + n * some large number]. A number of
    # things that happen next will depend on this seed, so we want to set it first.
    _ = Utils.set_seed(args.seed + Utils.rank_times_many(), verbose=False, device=device)

    ##################################################################################
    # Load any prior state. The arguments saved in prior state are used, but are
    # overridden by those currently in [args]. Observe how we use 'map_location'.
    ##################################################################################
    state = Utils.get_resume_file(args=args, save_folder=get_save_folder(args))
    if state is None:
        cur_epoch, train_step, state = 0, 0, None
    else:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        args = argparse.Namespace(**vars(state["args"]) | vars(args))
        cur_epoch, train_step = state["epoch"], state["train_step"]
        
    ##################################################################################
    # Construct dataset and DataLoader. Pay attention to:
    # 1. The training augmentations are stochastic, so they will depend on seeding
    # 2. The order of the training data is stochastic, so it will depend on seeding
    # 3. The length of both DataLoaders depends on the *per-GPU* batch size. We will
    #   need to adjust the number of steps per epoch accordingly
    ##################################################################################
    transform_tr = v2.Compose([v2.PILToTensor(),
        v2.RandomResizedCrop(size=(args.img_size, args.img_size), scale=(0.8, 1.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(min_max_normalization)])
    transform_val = v2.Compose([v2.PILToTensor(),
        v2.CenterCrop(size=(args.img_size, args.img_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(min_max_normalization)])
    
    data_tr = MNIST(root=".", train=True, transform=transform_tr)
    data_val = MNIST(root=".", train=False, transform=transform_val)
    
    sampler_tr, loader_tr = get_loader(data_tr, args=args, shuffle=True, drop_last=True)
    _, loader_val = get_loader(data_val, args=args, shuffle=False)

    # If we're not using gradient accumulation, setting [drop_last=True] ensures that
    # each epoch has the same number of gradient steps. However, if we are using
    # gradient accumulation, it can contain extra  batches at the end. Simulating
    # [drop_last=True] can be done by skipping these extra batches can be done by
    # ending each epoch early after a canonical number of gradient steps.
    steps_per_epoch = len(loader_tr) // args.accum_iter

    ##################################################################################
    # Model, optimizer, scheduler, gradient scaler, and learning rate scheduler.
    # It's important to construct the optimizer after applying nn.DataParallel or
    # nn.DistributedDataParallel in Utils.setup_model().
    ##################################################################################
    model = IMLEWrapper(args=args, model=MaskedAutoencoder(args))

    model = Utils.setup_model(model, args, state_dict=None if state is None else state["model"])
    scaler = torch.GradScaler(enabled=args.autocast)
    _ = None if state is None else scaler.load_state_dict(state["scaler"])
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    _ = None if state is None else optimizer.load_state_dict(state["optimizer"])
    scheduler = LinearRampThenCosineDecay(optimizer, args=args, steps_per_epoch=steps_per_epoch, last_epoch=train_step)

    ##################################################################################
    # Do some I/O now that training is about to start.
    ##################################################################################
    # twrite() is defined in Utils and is a smarter version of tqdm.write(), optimized
    # for possibly-DDP environments. Most DDP codebases instead modify the print()
    # function
    twrite(f"Beginning training for run={get_experiment_name(args)}")
    twrite(sorted([f"{k}={v}" for k,v in vars(args).items()]))

    # If we're using SLURM, there's a chance the job will get preempted or run out of
    # time before the experiment finishes. While the job scripts generated by
    # SlurmSubmit.py will automatically requeue in these cases, we DON'T want this to
    # happen if there were some error in the code. Writing a WandB init attempt file
    # indicates that nothing crashed prior to training starting, and is a necessary
    # (though insufficient) indicating that the job should requeue.
    if args.save_iter and Utils.dist_barrier_then_main():
        _ = UtilsSlurm.write_wandb_attempt(get_save_folder(args, make_folder=True))
    Utils.dist_barrier()
    
    ##################################################################################
    # Training loop. Immediately before it starts, each process sets the seed to
    # something determined by (1) the seed in [args], (2) the index of epoch about to
    # be run, and (3) its rank. Doing this immediately after saving state ensures that
    # we can reproduce the randomness used by subsequent training exactly.
    ##################################################################################
    _ = Utils.set_seed(cur_epoch + args.seed + Utils.rank_times_many(), verbose=True, device=device)
    last_save_time = time.time()

    # The [position] argument for TQDM is important as it prevents spurious prints.
    # position should increase with loop innerness.
    for epoch in tqdm_wrap(range(cur_epoch, args.epochs), desc="Epochs", leave=True, position=0):

        # This sets the seed for all workers to reflect the current epoch
        sampler_tr.set_epoch(epoch)

        # Just for the specific cIMLE implemenation we're using. But since [model]
        # could be a plain neural network, an nn.DataParallel model, a
        # nn.DistributedDataParallel model, or something weirder yet, we need one
        # function to unwrap it to the base plain neural network model.
        if (epoch+1) % args.sampler_update_iter == 0:
            _ = Utils.as_plain_nn(model).update_sampler()

        ##############################################################################
        # Demonstrates when we have determinism of randomness (not necessarily
        # computation). Try varying things like --num_workers, --accum_iter, and
        # --gpus to see how they impact randomness
        ##############################################################################
        random_rand = random.random()
        numpy_rand = np.random.rand()
        torch_rand = torch.rand(1).item()
        torch_cuda_rand = torch.rand(1, device=device).item()

        # Comment this out to make printed results nicer
        # twrite(f"Epoch {epoch+1:5}/{args.epochs} : random={random_rand:.4f}, numpy={numpy_rand:.4f}, torch={torch_rand:.4f}, torch_cuda={torch_cuda_rand:.4f}", all_procs=True)
        # Utils.dist_barrier()

        for idx,(x,y) in tqdm_wrap(enumerate(loader_tr), desc="Batches", leave=False, total=steps_per_epoch, position=1):

            # If the batch size used in the data loader is smaller than the desired
            # batch size (eg. for gradient accumulation), skip the last accum_iter-1
            # steps in each epoch as they wouldn't have been in full batches
            if idx >= steps_per_epoch * args.accum_iter:
                break

            # The scheduler steps prior to each gradient step
            if idx % args.accum_iter == 0:
                scheduler.step(train_step)

            # MOST IMPORTANT PART OF THE TRAINING LOOP. There three things to know:
            # 1. Utils.no_sync_context() is a no-op for non-DDP models, and equivalent
            #   to the no_sync() function in nn.DistributedDataParallel models. This
            #   disables gradient synchronization across processes, which isn't needed
            #   unless they're going to take a gradient step
            # 2. There's an implicit barrier before each backward pass, meaning that
            #    no process can get ahead of another. This doens't mean they are
            #   perfectly concurrent of course.
            # 3. Gradients are accumulated across processes via averaging, but across
            #   gradient accumulation steps by summing. This means that each forward
            #   pass computes the averaged-over-bs_per_pass gradient. To have the
            #   magnitude of the accumulated gradient match that of the gradient we'd
            #   have computed without gradient accumulation, we need to divide by the
            #   number of gradient accumulation steps
            with Utils.no_sync_context(model):
                with torch.autocast("cuda", enabled=args.autocast):
                    loss = model(x.to(device, non_blocking=True)).mean()
                scaler.scale(loss / args.accum_iter).backward()

            # The optimizer only steps when the accumulated gradient is ready. Thus
            # [train_step] is the number of times the optimizer has stepped
            if (idx+1) % args.accum_iter == 0:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad(set_to_none=True)
                train_step += 1

        ##############################################################################
        # EVALUATION IF NEEDED. Observe:
        # 1. We use a barrier to ensure that no process can enter evaluate() before
        #   others have finished updating their weights.
        # 2. All processes produce the same results because we enforce this in
        #   evaluate(), so we don't need to handle it here.
        # 3. Only the main process logs the results; the others wait at a barrier
        ##############################################################################
        results = argparse.Namespace(epoch=epoch+1, loss_tr=loss.item(), lr=scheduler.get_last_lr())

        if args.eval_iter and ((epoch+1) % args.eval_iter == 0 or epoch+1 == args.epochs):
            Utils.dist_barrier()

            # This function returns [results] after adding some extra fields
            results = evaluate(results=results, model=model, loader=loader_val, args=args)

            # Technically we could just use is_main_process() instead of
            # dist_barrier_then_main() since forcing different processes to return the
            # same results should synchronize them. However, it's better to spam
            # barriers where it won't meaningfully slow down training than to be sorry
            if Utils.dist_barrier_then_main():
                # _ = some function that logs results
                pass
            Utils.dist_barrier()
            twrite(f"Epoch {epoch+1:5}/{args.epochs} : lr={results.lr:.2e}, loss_tr={results.loss_tr:.4e} loss_val={results.loss_val:.4e}")
        else:
            twrite(f"Epoch {epoch+1:5}/{args.epochs} : lr={results.lr:.2e}, loss_tr={results.loss_tr:.4e}")

        ##############################################################################
        # SAVE STATE IF NEEDED. Observe:
        # 1. All processes should agree on the value of [normal_save] because it's
        #   determined only by the code being run
        # 2. Different processes could disagree whether enough time has passed to save
        #   again. To resolve this, we use (arbitrarily) the largest amount of elapsed
        #   time across all processes
        # 3. Only the main process actually needs to save anything. The other
        #   processes wait, and can not proceed to due to a barrier
        # 4. If we save, the very last thing that happens is reseeding training using
        #  the index of the next epoch. This means that if we resume training with
        #   this next epoch, we know how to set the seeds to exactly reproduce what it
        #   would've been
        ##############################################################################
        normal_save = args.save_iter and ((epoch+1) % args.save_iter == 0 or epoch+1 == args.epochs)
        time_since_last_save = Utils.all_reduce_max(time.time() - last_save_time)
        save_latest = args.save_iter and time_since_last_save > args.save_iter_t * 60

        if args.save_iter and (normal_save or save_latest):
            if Utils.dist_barrier_then_main():
                Utils.save_all(folder=get_save_folder(args), args=args,
                    model=model, optimizer=optimizer, scaler=scaler, epoch=epoch+1,
                    train_step=train_step, save_latest=save_latest and not normal_save)
            Utils.dist_barrier()
            last_save_time = time.time()
            _ = Utils.set_seed(epoch+1 + args.seed + Utils.rank_times_many(), verbose=True, device=device)
        
               












