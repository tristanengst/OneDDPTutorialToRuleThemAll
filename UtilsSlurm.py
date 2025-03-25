"""File containing functions to make chunking SLURM runs easier.

All imports are available in a plain Python environment.

"""
import argparse
from datetime import datetime, timedelta
import json
import os
import os.path as osp
import subprocess
import sys
import tarfile
import time

import UtilsPersistedState

#### Cluster information finding #####################################################
def get_cluster_type():
    """Returns a string for special host types, or None if they are not recognized."""
    h = os.uname()[1]
    if h.startswith("narval") or h.startswith("ng"):
        return "narval"
    elif h.startswith("cedar") or h.startswith("cdr"):
        return "cedar"
    elif h.startswith("beluga") or h.startswith("bg"):
        return "beluga"
    elif h.startswith("gra-") or h.startswith("gra") or h.startswith("gr"):
        return "graham"
    elif h.startswith("cs-star") or h.startswith("cs-venus"):
        return "solar"
    elif h.startswith("cs-apex"):
        return os.environ.get("CLUSTER_TYPE", "cs-apex")
    else:
        return None

def is_solar(): return get_cluster_type() == "solar"
def is_cc(): return get_cluster_type() in ["narval", "cedar", "beluga", "graham"]
def is_workstation(): os.uname()[1].startswith("cs-apex")

cluster2misc_reqs = dict(
    narval=dict(wandb_default_mode="online", default_account="rrg-keli"),
    cedar=dict(wandb_default_mode="online", default_account="rrg-keli"),
    beluga=dict(wandb_default_mode="online", default_account="def-keli"),
    graham=dict(wandb_default_mode="async", default_account="def-keli"),
    solar=dict(wandb_default_mode="online", default_account="debug"), # For now
    cs_apex=dict(wandb_default_mode="online", default_account=""))

# Specifies configuration for possible nodes/types of nodes, grouped by cluster.
# Commented out lines are for nodes not known to the scheduler.
#
# cpus_per_gpu  -- number of CPUs per GPU
# mem_per_gpu   -- amount of memory per GPU
# gpu_type      -- type of GPU (what we call it)
# gpu_name      -- type of GPU (what the scheduler calls it)
# gpus_per_node -- number of GPUs per node
# can_allocate  -- whether the node can be allocated
# max_time      -- maximum time in hours that can be requested
# constraint    -- constraint to use for the scheduler if possible
cluster2node2config = dict(
    solar={
        "cs-venus-01": dict(cpus_per_gpu=5, mem_per_gpu=84, gpu_type="q6000", gpus_per_node=6, can_allocate=True, gpu_name="quadro_rtx_6000"),
        # "cs-venus-02": dict(cpus_per_gpu=4, mem_per_gpu=64, gpu_type="2080", gpus_per_node=8, can_allocate=True, gpu_name="2080_ti"),
        "cs-venus-03": dict(cpus_per_gpu=4, mem_per_gpu=64, gpu_type="2080", gpus_per_node=4, can_allocate=True, gpu_name="2080_ti"),
        # "cs-venus-04": dict(cpus_per_gpu=6, mem_per_gpu=64, gpu_type="q4000", gpus_per_node=8, can_allocate=False, gpu_name="quadro_rtx_4000"),
        "cs-venus-05": dict(cpus_per_gpu=8, mem_per_gpu=60, gpu_type="a5000", gpus_per_node=8, can_allocate=True, gpu_name="rtx_a5000"),
        "cs-venus-06": dict(cpus_per_gpu=8, mem_per_gpu=60, gpu_type="a5000", gpus_per_node=8, can_allocate=True, gpu_name="rtx_a5000"),
        "cs-venus-07": dict(cpus_per_gpu=8, mem_per_gpu=128, gpu_type="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40"),
        "cs-venus-08": dict(cpus_per_gpu=8, mem_per_gpu=128, gpu_type="a100", gpus_per_node=4, can_allocate=True, gpu_name="a100"),
        "cs-venus-09": dict(cpus_per_gpu=7, mem_per_gpu=128, gpu_type="a40", gpus_per_node=8, can_allocate=True, gpu_name="a40"),
        # "cs-venus-10": dict(cpus_per_gpu=4, mem_per_gpu=64, gpu_type="a40", gpus_per_node=8, can_allocate=False, gpu_name="a40"),
        # "cs-venus-11": dict(cpus_per_gpu=4, mem_per_gpu=64, gpu_type="a40", gpus_per_node=8, can_allocate=False, gpu_name="a40"),
        "cs-venus-12": dict(cpus_per_gpu=10, mem_per_gpu=128, gpu_type="a6000", gpus_per_node=2, can_allocate=True, gpu_name="rtx_a6000"),
        "cs-venus-13": dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_type="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40"),
        "cs-venus-14": dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_type="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40"),
        # "cs-venus-15": dict(cpus_per_gpu=16, mem_per_gpu=240, gpu_type="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s"),
        "cs-venus-16": dict(cpus_per_gpu=16, mem_per_gpu=240, gpu_type="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s"),
        "cs-venus-17": dict(cpus_per_gpu=16, mem_per_gpu=240, gpu_type="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s"),
        "cs-venus-18": dict(cpus_per_gpu=16, mem_per_gpu=240, gpu_type="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s")},
    graham={
        "a5000": dict(cpus_per_gpu=16, mem_per_gpu=31, gpu_type="a5000", gpus_per_node=4, can_allocate=True, max_time=3, gpu_name="a5000", constraint="a5000"),
        "a100": dict(cpus_per_gpu=8, mem_per_gpu=63, gpu_type="a100", gpus_per_node=4, can_allocate=True, max_time=3, gpu_name="a100", constraint="a100"),
        "v100": dict(cpus_per_gpu=3, mem_per_gpu=23, gpu_type="v100", gpus_per_node=8, can_allocate=True, gpu_name="v100", constraint="v100&skylake"),
        "v100l": dict(cpus_per_gpu=5, mem_per_gpu=47, gpu_type="v100l", gpus_per_node=8, can_allocate=True, gpu_name="v100", constraint="v100&cascade")},
    beluga={"default": dict(cpus_per_gpu=10, mem_per_gpu=46, gpu_type="v100", gpus_per_node=4, can_allocate=True)},
    cedar={
        "p100": dict(cpus_per_gpu=6, mem_per_gpu=30, gpu_type="p100", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'")),
        "p100l": dict(cpus_per_gpu=6, mem_per_gpu=56, gpu_type="p100l", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'")),
        "v100l": dict(cpus_per_gpu=8, mem_per_gpu=46, gpu_type="v100l", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"))},
    narval={"a100": dict(cpus_per_gpu=12, mem_per_gpu=120, gpu_type="a100", gpus_per_node=4, can_allocate=True),
        "a100_3g.20gb": dict(cpus_per_gpu=6, mem_per_gpu=60, gpu_type="a100_3g.20gb", gpus_per_node=8, can_allocate=True),
        "a100_4g.20gb": dict(cpus_per_gpu=6, mem_per_gpu=60, gpu_type="a100_4g.20gb", gpus_per_node=8, can_allocate=True),},
    cs_apex={"default": dict(cpus_per_gpu=8, mem_per_gpu=48, gpu_type="3090", gpus_per_node=2, can_allocate=True)})


# Maps GPU type to the amount of VRAM it has
gpu2vram = {"2080": 8, "3090": 24, "a100_3g.20gb": 20, "a100_4g.20gb": 20} | dict(
    p100=12, p100l=16,
    q4000=8, q6000=24,
    v100=16, v100l=32,
    a5000=24, a6000=48, a40=48, l40s=48, a100=80 if is_solar() else 40)

cluster2node_prefix = dict(narval="ng", cedar="cdr", beluga="bg", graham="gra", solar="cs-venus", cs_apex="cs-apex")

### I/O UTILITY FUNCTIONS ############################################################
def twrite(*args, time=True):
    """Improved print statement. Unlike in most of the code, not TQDM-backed, but we
    keep the same name so that 'twrite'='smart print'.
    """
    time = f"[{datetime.now().isoformat(sep=' ', timespec='seconds')} {os.uname()[1]}]" if time else ""
    print(f"{time} {' '.join([str(x) for x in args])}")

### TASK STATUS MANAGEMENT ###########################################################
def write_wandb_attempt(model_folder):
    """Documents an attempt at initializing WandB."""
    _ = os.makedirs(model_folder, exist_ok=True)
    with open(f"{model_folder}/wandb_attempt.txt", "w+") as f:
        f.write(f"Attempted WandB init")
        twrite(f"Wrote wandb_init_attempt to {model_folder}")

def time_str_to_seconds(time_str):
    """Returns the number of seconds in a time string."""
    if "-" in time_str:
        days, hours_mins_secs = time_str.split("-")
    else:
        days, hours_mins_secs = 0, time_str

    secs_mins_hours_days = list(reversed(hours_mins_secs.split(":"))) + [days]
    mul_to_secs = [1, 60, 3600, 86400]
    return sum([int(t) * m for t,m in zip(secs_mins_hours_days, mul_to_secs)])

def time_str_to_minutes(time_str): return time_str_to_seconds(time_str) / 60
def time_str_to_hours(time_str): return time_str_to_seconds(time_str) / 3600

def time_stamp_to_date(time_stamp): return datetime.fromtimestamp(time_stamp).strftime("%Y %b %d %H:%M:%S")

def pretty_time_str(time_str):
    """Returns XXHYYM for XX hours and YY minutes. Days are collapsed to hours."""
    s = time_str_to_seconds(time_str)
    h = s // 3600
    m = (s % 3600) // 60
    h = str(h).zfill(max(2, len(str(h))+1))
    return f"{h}H{m:02d}M"

### SLURM UTILITY FUNCTIONS ##########################################################
def get_slurm_data(time=False):
    """Returns a Namespace describing the current SLURM job/task. If [time], then time
    information is included, but the call to squeue might be somewhat slow.
    """
    def get_job_data():
        """Returns a dictionary describing the current SLURM job/task."""
        return dict(job_id =int(os.environ.get("SLURM_JOB_ID", 0)),
            job_name=os.environ.get("SLURM_JOB_NAME", "1701"),
            gpus=int(os.environ.get("SLURM_GPUS_ON_NODE", 1)),
            node_list=os.environ.get("SLURM_JOB_NODELIST", []),
            account=os.environ.get("SLURM_JOB_ACCOUNT", "def-keli_gpu"),
            num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
            node_idx=int(os.environ.get("SLURM_NODEID", 0)),
            slurm_tmpdir=os.environ.get("SLURM_TMPDIR", "./"))

    def get_time_data():
        """Returns a dictionary with time information about the current job."""
        def get_time_allocated():
            """Returns the time limit in the current job/task as formatted by SLURM."""
            result = subprocess.getoutput("squeue -h -j $SLURM_JOB_ID -O TimeLimit") if "SLURM_JOB_ID" in os.environ else "168:00:00"
            return "168:00:00" if "slurm_load_jobs error:" in result else result.strip()

        def get_time_left():
            """Returns the time left in the current job/task as formatted by SLURM."""
            result = subprocess.getoutput("squeue -h -j $SLURM_JOB_ID -O TimeLeft") if "SLURM_JOB_ID" in os.environ else "168:00:00"
            return "168:00:00" if "slurm_load_jobs error:" in result else result.strip()
        
        return dict(time_left_hours=time_str_to_hours(get_time_left()),
            time_left_minutes=time_str_to_minutes(get_time_left()),
            time_limit=get_time_allocated(),
            time_limit_pretty=pretty_time_str(get_time_allocated()))

    result = get_job_data() | (get_time_data() if time else dict())
    return argparse.Namespace(**result)

def is_slurm():
    """Returns if the job is running in a SLURM environment."""
    return "SLURM_JOB_ID" in os.environ

def job_str():
    """Returns a string that describes the current job/task."""
    return f"JobID={os.environ['SLURM_JOB_ID']}" if is_slurm() else "Non-SLURM job"
    
def get_ordered_checkpoints(model_folder, highest_file=None, prefix=""):
    """Returns checkpoints in [model_folder] ordered from most to least recent.
    
    Args:
    model_folder    -- folder to search under
    highest_file    -- maximimally ordered checkpoint to include
    prefix          -- only consider files starting with this prefix
    """
    def file_to_checkpoint_order(f):
        f_mod = os.path.basename(f)
        f_mod = f_mod.replace(".pt", "").replace("_latest", "")
        f_mod = f_mod.lstrip(prefix).lstrip("-").lstrip("_")
        return int(f_mod) if f_mod.isnumeric() else None

    model_folder = osp.dirname(model_folder) if model_folder.endswith(".pt") else model_folder
    if osp.exists(model_folder):
        file2idx = {f"{model_folder}/{f}": file_to_checkpoint_order(f) for f in os.listdir(model_folder) if f.startswith(prefix) and f.endswith(".pt")}
        file2idx = {f: idx for f,idx in file2idx.items() if not idx is None}
        highest_idx = float("inf") if highest_file is None else file_to_checkpoint_order(highest_file)
        file2idx = {f: idx for f,idx in file2idx.items() if idx <= highest_idx}
        return sorted(file2idx.keys(), key=lambda x: file2idx[x], reverse=True)
    else:
        return []

### SLURM JOB ARRAY MANAGEMENT #######################################################
def is_finished(args):
    """Returns if the experiment given by --exp has marked itself as finished."""
    return osp.exists(args.exp) and osp.exists(f"{args.exp}/finished.txt")

def is_error(args):
    """Returns if the experiment seems to have an error. This is heuristic. Can be run
    before or after training, but if only run after, won't necessarily be run as prior
    code might time out.

    Args:
    args            -- Namespace with --exp attribute
    is_first_task   -- whether the current task is the first task in a chain of jobs
    before_training -- whether run before or after training
    """
    def check_code_tar_exists():
        """Returns if the code exists in the model folder."""
        result = osp.exists(f"{args.exp}/code.tar")
        if not result:
            twrite(f"CHECK_CODE_TAR_EXISTS(): check_code_tar_exists failed -> error : {args.exp}")
        return result

    def check_checkpoints_exist():
        prefix = "fn" if "finetunes" in args.exp else ""
        result = osp.exists(f"{args.exp}/wandb_data.pt") == bool(len(get_ordered_checkpoints(args.exp, prefix=prefix)))
        # result = True
        if not result:
            twrite(f"CHECK_CHECKPOINTS_EXIST(): check_checkpoints_exist failed -> error [{args.exp}]")
        return result

    def check_wandb_attempt():
        """This will ALWAYS return true at the start of a job. This is good if it's
        actually the first job in a chain, and otherwise, the check should've been
        made before submitting the current job by its parent.
        """
        result = args.before_training or osp.exists(f"{args.exp}/wandb_attempt.txt") 
        if not result:
            twrite(f"CHECK_WANDB_ATTEMPT(): check_wandb_attempt failed -> error (not in {args.exp})")
        return result

    return not (check_code_tar_exists() and check_checkpoints_exist() and check_wandb_attempt())

def cancel_self(time_to_wait=15):
    """Cancels the current job."""
    slurm_data = get_slurm_data()
    twrite(f"{job_str()} CANCEL_SELF(): Cancelling self in {time_to_wait} seconds with 'scancel {slurm_data.job_id}'")
    time.sleep(time_to_wait)
    subprocess.run(f"scancel {slurm_data.job_id}", shell=True, capture_output=True, text=True)

def resubmit(args=argparse.Namespace(), resubmit_due_to_err=False, **kwargs):
    """Resubmits the current job, and applies any updates to its account."""
    if UtilsPersistedState.persisted_state_get("already_resubmitted"):
        twrite(f"RESUBMIT(): Already resubmitted -> do nothing")
    elif UtilsPersistedState.persisted_state_get("resubmit_disabled"):
        twrite(f"RESUBMIT(): Resubmitting disabled -> do nothing")
    else:
        slurm_data = get_slurm_data()

        args = argparse.Namespace(**vars(args) | kwargs)
        args.slurm_script = args.slurm_script if "slurm_script" in args else UtilsPersistedState.persisted_state_get("slurm_script", default=lambda: twrite("RESUBMIT(): Fell back to finding 'slurm_script' in persisted state, but it wasn't there"))
        args.submit_dir = args.submit_dir if "submit_dir" in args else UtilsPersistedState.persisted_state_get("submit_dir", default=lambda: twrite("RESUBMIT(): Fell back to finding 'submit_dir' in persisted state, but it wasn't there"))
        assert not args.slurm_script is None, "RESUBMIT(): slurm_script is None"
        assert not args.submit_dir is None, "RESUBMIT(): submit_dir is None"
    
        twrite(f"RESUBMIT(): Resubmitting the job...")
        if not osp.exists(args.submit_dir):
            twrite(f"RESUBMIT(): Submit directory {args.submit_dir} does not exist -> will not resubmit")
            return

        if resubmit_due_to_err:
            twrite(f"RESUBMIT(): Excluding nodes {slurm_data.node_list}")
            extra_exclude_str = f"--exclude={slurm_data.node_list}"
        else:
            twrite(f"RESUBMIT(): Not excluding any nodes")
            extra_exclude_str = ""
        
        _ = UtilsPersistedState.persisted_state_update(already_resubmitted=True)
        resubmit_str = f"cd {args.submit_dir} ; sbatch -d afterany:{slurm_data.job_id} --parsable {args.slurm_script} {extra_exclude_str}"
        twrite(f"RESUBMIT(): {resubmit_str}")
        new_job_id = subprocess.run(resubmit_str, capture_output=True, shell=True, text=True)
        new_job_id = new_job_id.stdout.strip()

        # Currently doesn't work....
        # twrite(f"Putting job {new_job_id} on account {slurm_data.account}...")
        # _ = subprocess.run(f"scontrol update job {new_job_id} Account={slurm_data.account}")

        twrite(f"RESUBMIT(): Will continue run with job {new_job_id}")

def shutdown_check(args):
    if is_error(args):
        twrite(f"{job_str()} SHUTDOWN_CHECK(): found error -> ending")
        _ = cancel_self(time_to_wait=15) if is_slurm() else None
    elif is_finished(args):
        twrite(f"{job_str()} SHUTDOWN_CHECK(): found finished -> ending")
        _ = cancel_self(time_to_wait=15) if is_slurm() else None
    elif args.epi_time == 0:
        twrite(f"{job_str()} SHUTDOWN_CHECK(): found no error and not finished but epi time is zero -> ending")
    else:
        twrite(f"{job_str()} SHUTDOWN_CHECK(): found no error and not finished -> resubmit if on SLURM ({is_slurm()})")
        _ = resubmit(args) if is_slurm() else None

    if os.uname()[1].startswith("cs-venus"):
        _ = clean_old_slurm_tmpdirs()

def startup_check(args):
    """Handles the START of a task in a JobArray. For now, just make sure it hasn't
    errored or finished.
    """
    if is_error(args):
        twrite(f"{job_str()} STARTUP_CHECK(): found error -> ending")
        _ = cancel_self(time_to_wait=15)
    elif is_finished(args):
        twrite(f"{job_str()} STARTUP_CHECK(): found finished -> ending")
        _ = cancel_self(time_to_wait=15)
    else:
        twrite(f"{job_str()} STARTUP_CHECK(): found no error and not finished -> continue")

    # Save variables that might not be available later
    _ = UtilsPersistedState.persisted_state_update(**vars(args))

def need_to_tar_code(args):
    """Returns if a 'code.tar' file exists in the model folder."""
    return not osp.exists(f"{args.exp}/code.tar")

def get_time_limit(args):
    """Returns the time limit for the current training task. It must be less than the
    remaining time in the task; we will lose the last minute or so maybe.
    """
    slurm_data = get_slurm_data(time=True)
    delta = timedelta(minutes=slurm_data.time_left_minutes - args.epi_time)
    total_seconds = delta.days * 24 * 3600 + delta.seconds
    hours = total_seconds // (3600)
    minutes = (total_seconds - hours*3600) // 60
    seconds = total_seconds - hours*3600 - minutes*60
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def set_env_variables_and_return_existing(key2val):
    """Sets environment variable [k] to [v] for each key-value pair in [key2val].
    Returns a dictionary containing the keys of [key2val] with either their old values
    or '___SlurmUtilsDelete___' if they did not already exist. Keys whose values are
    '___SlurmUtilsDelete___' removed from the environment variables.
    """
    extant = {k: os.environ.get(k, default="___SlurmUtilsDelete___") for k in key2val}
    for k,v in key2val.items():
        if k == "___SlurmUtilsDelete___" and k in os.environ:
            del os.environ[k]
        else:
            os.environ[k] = v
    return extant

def set_job_to_cur_account(args):
    slurm_data = get_slurm_data()
    os.system(f"scontrol update job {args.job_id} Account={slurm_data.account}")

def copy_inside_parent(*, f, parent, force=-1, untar_if_tar=True, nodes=1, verbose=0):
    """Copies [f] inside [parent]. Only the basename of [f] will remain, ie.

    path/to/f -> parent/f

    Args:
    f               -- folder/file to copy
    parent          -- directory to place [f] under
    force           -- 1=always copy, 0=copy only if destination file doesn't exist,
                        -1=copy if src is newer than destination, or destination
                        doesn't exist
    untar_if_tar    -- untars [f] when copying                       
    """
    def f1_newer_than_f2(*, f1, f2):
        """Returns if file [f1] is newer than file [f2], or if [f2] doesn't exist."""
        return not osp.exists(f2) or osp.getmtime(f1) >= osp.getmtime(f2)
        
    def untar_under_parent(*, f, parent, exclude="", new_name=None):
        """Untars tarfile [f] under [parent]."""
        s = f"mkdir -p {parent} ; mkdir -p {parent} ; tar -xf {f} -C {parent} "
        twrite(f"COPY_INSIDE_PARENT(): Running {s}")
        os.system(s)

    def rsync_under_parent(*, f, parent, update=False, exclude="", new_name=None):
        update = "" if update else "-u"
        s = f"mkdir -p {parent} ; mkdir -p {parent} ; rsync -r {update} --info=progress2 {f} {parent}/ "
        twrite(f"COPY_INSIDE_PARENT(): Running {s}")
        os.system(s)

    if not osp.exists(f):
        raise ValueError(f"COPY_INSIDE_PARENT(): {f} does not exist")


    parent = parent.rstrip("/")
    src = f.rstrip(".tar") if untar_if_tar else f.rstrip("/")
    dest = f"{parent}/{osp.basename(f)}"
    dest = dest.rstrip(".tar") if untar_if_tar else dest

    if f.endswith(".tar") and untar_if_tar and force == 1:
        _ = untar_under_parent(f=f, parent=parent)
    elif f.endswith(".tar") and untar_if_tar and force == 0:
        if osp.exists(dest):
            twrite(f"COPY_INSIDE_PARENT(): found dest={dest}, not copying src={src} (force={force})")
        else:
            _ = untar_under_parent(f=f, parent=parent)
    elif f.endswith(".tar") and untar_if_tar and force == -1:
        if f1_newer_than_f2(f1=src, f2=dest):
            _ = untar_under_parent(f=f, parent=parent)
        else:
            twrite(f"COPY_INSIDE_PARENT(): found dest={dest} modified before src={src}, not copying (force={force})")
    elif force == 1:
        _ = rsync_under_parent(f=src, parent=parent, update=False)
    elif force == 0:
        if osp.exists(dest):
            twrite(f"COPY_INSIDE_PARENT(): found dest={dest}, not copying src={src} (force={force})")
        else:
            _ = rsync_under_parent(f=f, parent=parent, update=False)
    elif force == -1:
       _ = rsync_under_parent(f=f, parent=parent, update=True)

def get_slurm_tmpdir_folder(): return f"/localscratch/$USER/SLURM_TMPDIRS"

def get_running_jobs_list(username="tme3"):
    """Returns a list of running jobs by UID."""
    user = os.environ.get("USER")
    result = subprocess.getoutput(f"squeue -u {user} -O 'Name:500,State:.8'") if "SLURM_JOB_ID" in os.environ else "168:00:00"
    result = result.split("\n")[1:]
    job_names = [jn.split()[0] for jn in result]
    return job_names

def clean_old_slurm_tmpdirs():
    """On Solar, SLURM_TMPDIRS are of the form /localscratch/SLURM_TMPDIRS/EXP_NAME.
    
    Any that are at least 8 days old can/should be removed, or not running.
    """
    slurm_tmpdir = get_slurm_tmpdir_folder().replace("$USER", os.environ.get("USER"))
    eight_days_ago = datetime.now() - timedelta(days=8)
    running_jobs = sorted(get_running_jobs_list())

    twrite(f"Found running jobs list:" + '\n\t'.join(running_jobs))

    for f in os.listdir(slurm_tmpdir):
        filetime = datetime.fromtimestamp(osp.getctime(f"{slurm_tmpdir}/{f}"))
        if filetime < eight_days_ago:
            twrite(f"Removing {slurm_tmpdir}/{f} created at={time_stamp_to_date(osp.getctime(f'{slurm_tmpdir}/{f}'))}")
            os.system(f"rm -rf {slurm_tmpdir}/{f}")
        elif not any([osp.basename(f) in rj for rj in running_jobs]):
            twrite(f"Removing {slurm_tmpdir}/{f} created at={time_stamp_to_date(osp.getctime(f'{slurm_tmpdir}/{f}'))} since it was not found to be running")
            os.system(f"rm -rf {slurm_tmpdir}/{f}")
        else:
            pass

def setup_solar_python_env(env="py311OneDDPTutorial", cp_force=False):
    """
    """
    user = os.environ.get("USER")

    if osp.exists(f"/localscratch/$USER/{env}"):
        os.system(f"source /localscratch/$USER/{env}/bin/activate")
    elif osp.exists(f"/localscratch/$USER/{env}.tar.gz"):
        os.system(f"tar -xf /localscratch/$USER/{env}.tar.gz -C /localscratch/$USER/{env}")
        os.system(f"source /localscratch/$USER/{env}/bin/activate")
    elif osp.exists(f"/project/apex-lab/$USER/{env}.tar.gz"):
        os.system(f"tar -xf /project/apex-lab/$USER/{env}.tar.gz -C /localscratch/$USER/{env}")
        os.system(f"source /localscratch/$USER/{env}/bin/activate")

def get_git_sha_from_git_folder():
    return subprocess.run("git rev-parse HEAD", capture_output=True, shell=True, text=True).stdout.strip()

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--function", type=str, required=True, choices=["startup_check", "shutdown_check", "new_submit_if_needed", "need_to_tar_code", "get_time_limit", "write_to_tasks", "copy", "get_git_sha"],
        help="Functionality to implement")
    P.add_argument("--exp", type=str, 
        help="Path to folder where experiment results, checkpoints, etc. are saved")
    P.add_argument("--slurm_script", type=str, 
        help="Slurm script submitting the experiment (likely calling functions from this file)")
    P.add_argument("--before_training", action="store_true",
        help="Whether or not training has started in the current task")
    P.add_argument("--epi_time", default=15, type=int,
        help="Amount of time to the last task must end training early in minutes")
    P.add_argument("--env_vars", default=[], nargs="*",
        help="Environment variables to set/unset")
    P.add_argument("--submit_dir", default=osp.expanduser("~/Development/IMLE-SSL-2"),
        help="Submit directory")
    P.add_argument("--cp_src", help="File to copy and/or untar")
    P.add_argument("--cp_parent", help="Parent of destination for copied file")
    P.add_argument("--untar_if_tar", choices=[0, 1], default=1, type=int,
        help="Untars .tar files when copying")
    P.add_argument("--cp_force", choices=[0, 1, -1], default=0, type=int,
        help="1=always copy, 0=copy if dest doesn't exist, -1=copy if source is newer or dest doesn't exist")
    args = P.parse_args()    

    extant_env_vars = set_env_variables_and_return_existing(dict([tuple(e.split("=")) for e in args.env_vars]))

    if args.function == "startup_check":
        _ = startup_check(args)
    elif args.function == "shutdown_check":
        _ = shutdown_check(args)
    elif args.function == "new_submit_if_needed":
        _ = new_submit_if_needed(args)
    elif args.function == "need_to_tar_code":
        print("true" if need_to_tar_code(args) else "false")
    elif args.function == "get_time_limit":
        print(get_time_limit(args))
    elif args.function == "write_to_tasks":
        _ = write_to_tasks(args.exp)
    elif args.function == "need_to_resubmit":
        print("true" if need_to_resubmit(args) else "false")
    elif args.function == "copy":
        _ = copy_inside_parent(f=args.cp_src, parent=args.cp_parent, untar_if_tar=args.untar_if_tar, force=args.cp_force)
    elif args.function == "get_git_sha":
        print(get_git_sha_from_git_folder()) 
    else:
        raise ValueError(f"Unknown function: {args.function}")
    
    _ = set_env_variables_and_return_existing(extant_env_vars)
