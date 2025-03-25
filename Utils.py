import argparse
import contextlib
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
import os
import os.path as osp
import random
import sys
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing
from tqdm import tqdm


######################################################################################
# I/O related functions. These are nicer version of TQDM modified to be easily used
# in DDP contexts.
######################################################################################
def pretty_time(offset=False):
    offset = " " * 6 if offset else ""
    return f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]{offset}"

def pretty_time_rank(offset=False):
    offset = " " * 6 if offset else ""
    return f"[{datetime.now().isoformat(sep=' ', timespec='seconds')} host={os.uname()[1]} rank={get_rank()}]{offset}"

def tqdm_wrap(x, offset=True, disable=False, **kwargs):
    """Smarter version of tqdm() optimized for DDP."""
    if "desc" in kwargs:
        kwargs["desc"] = f"{pretty_time(offset=offset)} {kwargs['desc']}"
    defaults = dict(leave=False, dynamic_ncols=True, file=sys.stdout, delay=0, mininterval=1, ncols=30, disable=disable or not is_main_process())
    return tqdm(x, **defaults | kwargs)

def twrite(*args, all_procs=False, offset=False, time=True):
    """Smarter version of tqdm.write optimized for DDP."""
    if all_procs:
        l = [f"{pretty_time_rank(offset=offset)}"] + list(args) if time else list(args)
        tqdm.write(" ".join([str(x) for x in l]))
    elif is_main_process():
        l = [f"{pretty_time(offset=offset)}"] + list(args) if time else list(args)
        tqdm.write(" ".join([str(x) for x in l]))

######################################################################################
######################################################################################
######################################################################################



######################################################################################
# Utilities for managing DDP
######################################################################################


# Every single DDP-capable codebase you see will have a function with this name. It
# sets up the DDP environment, and in this case also modifies [args] to contain any
# new information
def init_distributed_mode(args, ):
    """Returns [args] with DDP-related arguments set. If DDP is being used, also
    initialize it.
    """
    # Set the URL that different DDP processes use to talk to each other. If it's the
    # TCP method, it should already be in the environment variables.
    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    else:
        dist_url = "env://"

    # NCCL is almost always the best choice, but sometimes only GLOO will work (eg.
    # Graham A5000 and A100 nodes)
    if args.hw_dist_backend == "adapt" and "ddp" in args.speedup:
        backend = "nccl"
    else:
        backend = args.hw_dist_backend

    assert torch.distributed.is_nccl_available()

    # Depending on whether or not we're using DDP, and the system we're doing it on,
    # set arguments for DDP usage differenty. The base case is not using DDP
    if not "ddp" in args.speedup:
        twrite("Not using distributed mode")
        return argparse.Namespace(**vars(args) | dict(distributed=False, gpu="cuda",
            world_size=1, rank=0, dist_url=dist_url))
   
    # Non-SLURM DDP settings, assumes single node
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        gpu = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        ddp_init_kwargs = dict(world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
            init_method=dist_url,
            backend=backend)
    
    # SLURM DDP settings, allows mult-node
    elif "SLURM_NODEID" in os.environ:
        gpus_per_node = torch.cuda.device_count()
        assert gpus_per_node == len(args.gpus)
        gpu = f"cuda:{int(os.environ.get('SLURM_LOCALID'))}",
        ddp_init_kwargs = dict(world_size=gpus_per_node * args.nodes,
            rank=int(os.environ.get("SLURM_NODEID")) * gpus_per_node + int(os.environ.get("SLURM_LOCALID")),
            init_method=dist_url,
            backend=backend)
    else:
        raise ValueError(f"Not in a distributed environment but speedup={args.speedup}")

    # Update [args] to match the DDP settings, then initialize DDP. The 'timeout'
    # argument to the DDP initialization is the amount of time before a process not
    # communicating with rest will cause a crash. It should be relatively large to
    # allow for writing large checkpoints to disk on ComputeCanada, where this takes a
    # high variable amount of time
    args = argparse.Namespace(**vars(args) | ddp_init_kwargs, distributed=True, gpu=gpu, dist_url=dist_url)

    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(
        device_id=torch.device(args.gpu),
        timeout=timedelta(seconds=4800),
        **ddp_init_kwargs)

    # No process gets to return until all processes have reached this point
    dist.barrier()
    return args

# These are the bread-and-butter functions for DDP. All are self-explanatory. I've
# include a number of less-standard utilities that are useful too
def is_dist_avail_and_initialized(): return dist.is_available() and dist.is_initialized()
def get_world_size(): return dist.get_world_size() if is_dist_avail_and_initialized() else 1
def get_rank(): return dist.get_rank() if is_dist_avail_and_initialized() else 0
def is_main_process(): return get_rank() == 0

def get_device(args):
    """Returns the device ON THE CURRENT NODE given [args] and CUDA availability."""
    if args.gpus == [-1] or not torch.cuda.is_available():
        return torch.device("cpu")
    elif args.speedup == "gpu" or args.speedup == "compile":
        return torch.device(f"cuda:{args.gpus[0]}")
    else:
        return torch.device(f"cuda:{get_rank() % torch.cuda.device_count()}")

def is_main_process_pre_init():
    """Like is_main_process(), but works before dist.init_process_group() is called."""
    return int(os.environ["RANK"]) == 0 if "RANK" in os.environ else True

def assert_main_process():
    assert is_main_process(), f"This function should only be called by the main process, but was called by rank {get_rank()}"

# Used frequently. No process can poass dist.barrier() until all others have reached it
def dist_barrier(name=None):
    if not name is None:
        twrite(f"Waiting at: {name}", all_procs=True)
    _ = dist.barrier() if is_dist_avail_and_initialized() else None
    if not name is None:
        twrite(f"Done waiting at: {name}", all_procs=True)
    return None

# A common pattern is to have a block of code that only the main process should run,
# and no process should continue from until the main process does too. So:
#
# if dist_barrier_then_main():
#     # main process does stuff
# Utils.dist_barrier()
def dist_barrier_then_main(name=None):
    dist_barrier(name=name)
    return is_main_process()

def from_main_process(x, group=None):
    """Returns [x] matching whatever [x] is in the main process. Synchronizes. Don't
    use with tensors. The key usage is forcing things set stochastically in a call to
    get_args() like WandB UIDs to match the main process.
    """
    if is_dist_avail_and_initialized():
        input_obj_list = [x] * get_world_size()
        output_obj_list = [x]
        dist.scatter_object_list(output_obj_list, input_obj_list, src=0, group=group)
        return output_obj_list[0]
    else:
        return x

def rank_times_many():
    """Returns the rank times a big number. Very useful in setting seeds."""
    return get_rank() * 65536

def all_reduce(x, device="cuda", op=dist.ReduceOp.SUM, **kwargs):
    """Returns the sum of [x] across all processes."""
    x = int(x) if isinstance(x, bool) else x
    x = torch.tensor(x, device=device) if not isinstance(x, torch.Tensor) else x
    _ = dist.all_reduce(x, op=op, **kwargs) if is_dist_avail_and_initialized() else None
    return x

def all_reduce_sum(x, device="cuda"):
    """Returns the sum of [x] across all processes."""
    x = int(x) if isinstance(x, bool) else x
    x = torch.tensor(x, device=device) if not isinstance(x, torch.Tensor) else x
    _ = dist.all_reduce(x, op=dist.ReduceOp.SUM) if is_dist_avail_and_initialized() else None
    return x

def all_reduce_max(x, device="cuda"):
    """Returns the max of [x] across all processes."""
    x = int(x) if isinstance(x, bool) else x
    x = torch.tensor(x, device=device) if not isinstance(x, torch.Tensor) else x
    _ = dist.all_reduce(x, op=dist.ReduceOp.MAX) if is_dist_avail_and_initialized() else None
    return x

def all_reduce_min(x, device="cuda"):
    """Returns the max of [x] across all processes."""
    x = int(x) if isinstance(x, bool) else x
    x = torch.tensor(x, device=device) if not isinstance(x, torch.Tensor) else x
    _ = dist.all_reduce(x, op=dist.ReduceOp.MIN) if is_dist_avail_and_initialized() else None
    return x

def all_gather_cat(x, dim=0):
    """Returns the list of [x] from all processes."""
    # Should work but doesn't. Maybe it's an issue with the PyTorch version?
    if is_dist_avail_and_initialized():
        all_x = [torch.zeros_like(x) for _ in range(get_world_size())]
        print(torch.distributed.get_backend(group=None))
        _ = dist.all_gather(all_x, x)
        return torch.cat(all_x, dim=dim)
    else:
        return x

def args_with_ddp_args_set(args):
    """Returns [args] with some adaption for the DDP setting at hand."""
    args.hw_mp_ctx = ("spawn" if "ddp" in args.speedup else None) if args.hw_mp_ctx == "adapt" else args.hw_mp_ctx
    args.dist_on_itp = True if "ddp" in args.speedup else False
    return args

def no_sync_context(model):
    """Returns the no_sync() context manager for [model] if it is DDP, or a context
    that does nothing.
    """
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.no_sync()
    else:
        return contextlib.nullcontext()

def as_plain_nn(model):
    """Returns the model without optimization wrappers."""
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        return as_plain_nn(model._orig_mod)
    elif isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        return as_plain_nn(model.module)
    elif isinstance(model, nn.DataParallel):
        return model.module
    else:
        return model

######################################################################################
######################################################################################
######################################################################################


######################################################################################
# Helps with controlling randomness.
# IGNORE IF YOU'RE JUST INTERESTED IN PURE DDP
######################################################################################
def set_seed(seed, *, device, verbose=True):
    """Seeds the program to use seed [seed] on device [device]."""
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # It's not 100% clear if this is needed given torch.cuda.manual_seed(), but
        # the docs think it might be.
        torch.cuda.manual_seed_all(seed)
        if verbose:
            twrite(f"Process={get_rank()}: Set the NumPy, PyTorch, and Random seeds to {seed}", all_procs=True)
    elif isinstance(seed, dict):
        random.setstate(seed["random_seed"])
        np.random.set_state(seed["numpy_seed"])
        torch.set_rng_state(seed["torch_seed"])
        torch.cuda.set_rng_state(seed["torch_cuda_seed"], device=device)
        if verbose:
            twrite(f"Set the NumPy, PyTorch, and Random modules seeds to old seed")
    else:
        raise ValueError(f"Seed should be int or contain resuming keys")

    return seed

def stateless(fn):
    """Decorator to do a function in a stateless way. Some things with eg.
    multiprocessing are surprisingly and annoyingly non-stateless! Note that seeds
    used deliberately inside the function can still change state! This function is
    intended to allow not worrying about this.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with SeedContextManager("cur_seed"):
            return fn(*args, **kwargs)
    return wrapper

class SeedContextManager:
    """Context manager for forcing all seeds to a value at the start of the context,
    and setting them to what they were prior to the context starting immediately upon
    its end.

    Args:
    seed    -- an integer seed or seed state dict, or None. In the latter case, using
                the SeedContextManager is a no-op
    """
    def __init__(self, seed=0, name="", device="cuda"):
        self.seed = get_seed_dict(device=device) if seed == "cur_seed" else seed
        self.old_seed_cur_state = None
        self.seed_cur_state = None
        self.name = name
        self.device = device

    def __enter__(self):
        if not self.seed is None:
            self.old_seed_cur_state = get_seed_dict(device=self.device)
            seed = self.seed if self.seed_cur_state is None else self.seed_cur_state
            _ = set_seed(seed, verbose=False, device=self.device)

    def __exit__(self, type, value, traceback):
        self.seed_cur_state = get_seed_dict(device=self.device)
        if not self.old_seed_cur_state is None:
            _ = set_seed(self.old_seed_cur_state, verbose=False, device=self.device)

    def __str__(self):
        return f"{self.__class__.__name__} [name={self.name}, cur_randbits={self.getrandbits_nochange()}]"

    def state_str_nochange(self, digits_per_lib=4):
        with self:
            return state_to_str(digits_per_lib=digits_per_lib)

    def getrandbits(self, k=32):
        """Seeded version fo random.getrandbits()"""
        with self:
            return random.getrandbits(k)

    def getrandbits_nochange(self, k=32):
        with self:
            with SeedContextManager("cur_seed"):
                return random.getrandbits(k)

    def state_dict(self): return dict(seed=self.seed,
        seed_cur_state=self.seed_cur_state,
        old_seed_cur_state=self.old_seed_cur_state)

    def load_state_dict(self, state_dict):
        state_dict = torch.load(state_dict, weights_only=False) if isinstance(state_dict, str) else state_dict
        if self.seed == state_dict["seed"]:
            self.seed_cur_state = state_dict["seed_cur_state"]
            self.old_seed_cur_state = state_dict["old_seed_cur_state"]
        else:
            raise ValueError(f"Can not load SeedContextManager state as the start seed {self.seed} does not match the seed of the SeedContextManager being loaded: {state_dict['seed']}")

def get_seed_dict(device="cuda"):
        """Returns a dictionary giving the seeds when the function is called."""
        return dict(random_seed=random.getstate(),
            torch_seed=torch.get_rng_state(),
            torch_cuda_seed=torch.cuda.get_rng_state(device=device),
            numpy_seed=np.random.get_state())

@stateless
def state_to_str(digits_per_lib=4, device="cuda"):
    """Returns a string representing the state of the random number generators."""
    multiplier = 10 ** digits_per_lib
    torch_int = int(torch.rand(1).item() * multiplier)
    torch_cuda_int = int(torch.rand(1, device=device).item() * multiplier)
    np_int = int(np.random.rand() * multiplier)
    rand_int = int(random.random() * multiplier)
    return f"device={device}={rand_int}.{np_int}.{torch_int}.{torch_cuda_int}"

######################################################################################
######################################################################################
######################################################################################

######################################################################################
# Model setup, saving and loading.
######################################################################################

def get_resume_file(*, args, save_folder):
    """Returns the file to resume from given [args] and [save_folder], or None if no
    such file is found.

    Args:
    args        -- argparse Namespace
    save_folder -- path to folder where checkpoints would be saved for the current run
    """
    # Try and find the file to resume from based on the UID
    if args.resume == "none":
        return None
    elif args.resume == "latest" and osp.exists(save_folder):
        checkpoints = [f for f in os.listdir(save_folder) if f.startswith("checkpoint_")]
        if len(checkpoints) == 0:
            return None

        sorted_checkpoints = sorted(checkpoints, key=lambda f: int(f.replace("_latest.pt", "").replace("checkpoint_", "").replace(".pt", "")))
        return osp.join(save_folder, sorted_checkpoints[-1])
    elif args.resume == "latest" and not osp.exists(save_folder):
        return None
    elif osp.exists(args.resume):
        return args.resume
    else:
        raise FileNotFoundError(f"Could not find file to resume from: {args.resume}")

# Previously, it was actually really important that this function was stateless.
@stateless
def setup_model(model, args=None, state_dict=None, speedup=None, device=None, gpus=None, ddp_kwargs=dict(), no_compile=False):
    """Sets up model [model] according to argparse Namespace [args]. The model should
    be on the CPU. This takes care moving the model to the right device and loading
    existing weights.

    Because we use as_plain_nn when saving, we don't need to worry about that.

    With DDP, use map_location correctly when loading the state_dict into memory:
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html.

    Speedup options:
    gpu             -- use one GPU
    dataparallel    -- use DataParallel over all specified GPUs
    compile         -- use one GPU and torch.compile
    ddp             -- use DDP over all specified GPUs
    compile_ddp     -- use DDP wrapper over compiled modules for all specified GPUs
    ddp_compile     -- compile the DDP wrapper over model for all specified GPUs.

    With two GPUs, the speed ordering is roughly

    gpu < compile < dataparallel < ddp < compile_ddp < ddp_compile

    According to According to discuss.pytorch.org/t/how-should-i-use-torch-compile-properly,
    'ddp_compile' might be faster but less robust than 'compile_ddp'.

    Args:
    model       -- model to set up
    args        -- argparse Namespace with the settings
    state_dict  -- state_dict to load into the model (must be given even if [args] specifies something)
    speedup     -- speedup method to use
    device      -- device to use if not given by [args]
    gpus        -- GPUs to use if not given by [args]
    ddp_kwargs  -- keyword arguments to pass to DistributedDataParallel
    no_compile  -- don't compile the model even if given by [speedup]
    """
    def load_state_dict_(model, state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k,v in state_dict.items()}
        incompatible_keys = model.load_state_dict(state_dict, strict=False)

        # Presumably you'd raise an error if there are any incompatible keys!
        return model

    speedup = args.speedup if speedup is None else speedup
    device = get_device(args) if device is None else device # In this case [args] can't be None
    gpus = args.gpus if gpus is None else gpus

    model = model.to("cpu")
    if speedup == "dataparallel":
        model = model if state_dict is None else load_state_dict_(model, state_dict)
        return nn.DataParallel(model, device_ids=gpus).to(device)
    elif speedup == "gpu":
        model = model if state_dict is None else load_state_dict_(model, state_dict)
        return model.to(device)
    elif speedup == "compile" and not no_compile:
        model = model.to(device)
        dist_barrier()
        model = model if state_dict is None else load_state_dict_(model, state_dict)
        model = torch.compile(model)
        dist_barrier()
        return model
    elif speedup == "ddp":
        dist_barrier()
        model = model if state_dict is None else load_state_dict_(model, state_dict)
        dist_barrier()
        return DistributedDataParallel(model.to(device), **ddp_kwargs)
    elif speedup == "compile_ddp" and not no_compile:
        dist_barrier()
        model = setup_model(model, args=args, state_dict=state_dict, speedup="compile",
            device=device, gpus=gpus)
        dist_barrier()
        return DistributedDataParallel(model, **ddp_kwargs)
    elif speedup == "ddp_compile":
        dist_barrier()
        model = setup_model(model, args=args, state_dict=state_dict, speedup="ddp",
            device=device, gpus=gpus, ddp_kwargs=ddp_kwargs)
        dist_barrier()
        return torch.compile(model.to(device))
    else:
        raise ValueError(f"Unknown speedup {speedup}")

def save_all(*, folder, args, save_latest=False, epoch=None, fname=None, prefix="checkpoint", verbose=False, **kwargs):
    """Saves all the state in [kwargs] along with [args].

    Exactly one of [epoch] and [fname] must be given.

    Args:
    folder      -- folder to save to
    save_latest -- save to 'FOLDER/checkpointEPOCH_latest.pt' and delete all other
                    'checkpointFOLDER/*_latest.pt' files.
    epoch       -- epoch number to save with. Gives folder/epoch.pt or folder/epoch_latest.pt.
    fname       -- filename to save to inside [folder]. Gives folder/fname.pt or folder/fname_latest.pt.
    kwargs      -- key-value pairs of things to save
    """
    assert is_main_process(), "Only the main process can save"
    def map_saved_by_type(x):
        if isinstance(x, nn.Module):
            return as_plain_nn(x).state_dict()
        elif hasattr(x, "state_dict"):
            return x.state_dict()
        else:
            return x

    assert not (epoch is None) == (fname is None), "Exactly one of epoch and fname must be given"
    fname = f"{prefix}{epoch}" if fname is None else fname.rstrip(".pt")

    state_dict = dict(epoch=epoch, args=args, **{k: map_saved_by_type(v) for k,v in kwargs.items()})
    _ = os.makedirs(folder, exist_ok=True)

    def prefix_matches_file(*, prefix, fname):
        if prefix == "":
            return osp.basename(fname)[0].isnumeric()
        else:
            return osp.basename(fname).startswith(prefix)

    if save_latest:
        latest_files = [f for f in os.listdir(folder) if f.endswith("_latest.pt") and prefix_matches_file(prefix=prefix, fname=f)]
        if len(latest_files) <= 1:
            for f in latest_files:
                os.remove(f"{folder}/{f}")
        else:
            raise ValueError(f"More than one '_latest.pt' file in {folder}. This is a bug. Got: {latest_files} for prefix='{prefix}'")
        save_to_file = f"{folder}/{fname}_latest.pt"
    else:
        save_to_file = f"{folder}/{fname}.pt"

    save_start_time = time.time()
    _ = torch.save(state_dict, save_to_file)
    save_end_time = time.time()

    del state_dict

    if verbose:
        twrite(f"Saving (elapsed time={(save_end_time - save_start_time)/60:.2f}m): {save_to_file}")
