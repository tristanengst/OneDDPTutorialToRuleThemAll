# One DDP Tutorial to Rule Them All
While many DDP tutorials exist, the reality is that DDP remains a chaotic mess where things that should work don't. Most tutorials lack the important information and utilities that bring order to this chaos and provide crucial quality-of-life improvements. This one rectifies these issues with an explicit focus on APEX lab compute resources. *Key features*:

1. Automatic batch size scaling
2. Automatically sets good defaults you didn't know needed to be set
3. Handles randomness as well as possible
4. Works on Solar and ComputeCanada SLURM environments without changing anything. On ComputeCanada, multi-node training happens by specifing `--nodes X` with `X` more than one
5. TBD: SafeTensors data—handles arbitrary image datasets _without_ changing the data distribution. Not only 20% faster, but also fixes a yet-unidentified issue with JPEG data 


### Setup
```
git clone https://github.com/tristanengst/OneDDPTutorialToRuleThemAll
cd OneDDPTutorialToRuleThemAll
conda create -n py311OneDDPTutorial python=3.11
conda activate py311OneDDPTutorial
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install einops wandb # WandB account not needed, just the package
python -c "from torchvision.datasets import MNIST ; m = MNIST(root='.', download=True)"
```

### Files
To get a sense and reasonable understanding of DDP, look here—you can skip to the `get_args()` function (about line 200):
- `TrainAndEval.py`

And at `init_distributed_mode()` here:
- `Utils.py`

I've included other files as a minimal working example of how to make managing many jobs that use lots of GPUs tractable, especially on SLURM:
- `pip_reqs_cc.txt` would be useful if we were running on ComputeCanada
- `SlurmSubmit.py` is a large, complicated script that makes submitting jobs non-annoying
- `UtilsSlurm.py` is used by `SlurmSubmit.py`
- `UtilsPersistedState.py` is used by `UtilsSlurm.py`
Also see [ScriptsAndAliases](https://github.com/tristanengst/ScriptsAndAliases) for this.

### Workstation Usage Examples
There are no config files; every input to an experiment must be set on the command line. DDP with GPUs `0` and `1`:
```
torchrun --standalone --nnodes=1 --nproc-per-node NUMBER_OF_GPUS TrainAndEval.py --gpus 0 1 --speedup ddp ...
```
DDP with GPUs `3` and `4` and compilation:
```
torchrun --standalone --nnodes=1 --nproc-per-node NUMBER_OF_GPUS TrainAndEval.py --gpus 3 4 --speedup compile_ddp ...
```
DataParallel on two GPUs:
```
torchrun --standalone --nnodes=1 --nproc-per-node NUMBER_OF_GPUS TrainAndEval.py --gpus 0 1 --speedup dataparallel ...

```
Single GPU, compilation, DDP:
```
torchrun --standalone --nnodes=1 --nproc-per-node 1 TrainAndEval.py --gpus 1 --speedup compile_ddp ...

```
etc.

See the `get_args()` function in `TrainAndEval.py` for further arguments.

### SLURM Usage Examples
Single-node:
```
python SlurmSubmit.py TrainAndEval.py --speedup compile_ddp --gpus 0 1 2 3 ... --time 12:00:00 --account cs-gpu-research
```
Multi-node:
```
python SlurmSubmit.py TrainAndEval.py --speedup compile_ddp --gpus 0 1 2 3 --nodes 2 ... --time 6:00:00 --account def-keli
```