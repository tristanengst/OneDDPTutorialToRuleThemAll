import argparse
import json
import os
import os.path as osp
import time
import random
from tqdm import tqdm
import tarfile
import Utils

import UtilsSlurm
from UtilsSlurm import get_cluster_type, is_solar, is_cc, cluster2node2config, cluster2misc_reqs, cluster2node_prefix, gpu2vram

######################################################################################
# Really miscellaneous utilities
######################################################################################
def args_with_sf_data_set(args, keys=["data_tr", "data_val"]):
    """Returns [args] with each dataset in [keys] set to use a safetensors file if one
    is available and --data_allow_sf is set. Otherwise, returns [args] unchanged.
    """
    extn2sf_extn = {"tar": ".tar", "lmdb": "", "": ""}

    if args.data_allow_sf:
        non_sf_keys = [k for k in keys if not "safetensors" in vars(args)[k]]
        for k in non_sf_keys:
            fname, extn = vars(args)[k].split(".")
            possible_sf = f"{fname}_safetensors{extn2sf_extn[extn]}"
            if osp.exists(possible_sf):
                tqdm.write(f"------ Changing to use safetensors: {k}={vars(args)[k]} -> {k}={possible_sf}")
                vars(args)[k] = possible_sf
            else:
                tqdm.write(f"------ Could not find safetensors file {possible_sf} for {k}={vars(args)[k]}. No change.")
        return argparse.Namespace(**vars(args))
    else:
        return args

def is_tarfile(f):
    """Returns if [f] is a .tar file"""
    return f.endswith(".tar") or f.endswith(".tar.gz") or f.endswith(".tgz")

def get_node_reqs(*, args, slurm_args, distributed=False):
    """Returns a (gpus_per_node, total_gpus, cpus_per_gpu, mem, exluded_nodes) dictionary.

    Args:
    args        -- arguments
    slurm_args  -- slurm submission arguments
    distributed -- whether the run will use DDP or not
    """
    nodes = args.nodes
    gpus_per_node = len(args.gpus)
    total_gpus = gpus_per_node * args.nodes

    if get_cluster_type() == "solar":
        # On Solar, we need to find all nodes that support the requested config, and
        # exclude the rest, together with any nodes we've explicitly excluded too
        node2config = cluster2node2config["solar"]
        excluded_nodes = [n for n in slurm_args.exclude if n.startswith(cluster2node_prefix[get_cluster_type()])]

        can_allocate = lambda n: node2config[n]["can_allocate"]
        enough_gpus = lambda n: node2config[n]["gpus_per_node"] >= gpus_per_node
        enough_cpus = lambda n: node2config[n]["cpus_per_gpu"] * node2config[n]["gpus_per_node"] >= (args.num_workers * gpus_per_node)
        enough_ram = lambda n: slurm_args.mem == "adapt" or node2config[n]["mem_per_gpu"] * gpus_per_node >= int(slurm_args.mem.replace("G", ""))
        
        if slurm_args.gpu_type == "adapt":
            slurm_args.gpu_type = "a100,a40,a5000,l40s,a6000,q6000,2080"
        else:
            pass

        # Filter out nodes who can't be allocated or 
        included_nodes = [n for n in node2config if node2config[n]["gpu_type"] in slurm_args.gpu_type.split(",")
            and can_allocate(n) and enough_gpus(n) and enough_cpus(n) and enough_ram(n)]
        gpu_type = None

        # Sanity check to make sure all requested GPUs are valid. Exluded nodes
        # haven't been accounted for yet
        included_gpu_types = {node2config[n]["gpu_type"] for n in included_nodes}

        print(f"Included nodes:", included_nodes)
        # if not ",".join(sorted(included_gpu_types)) == ",".join(sorted(slurm_args.gpu_type.split(","))):
        #     raise ValueError(f"Requested gpu types {slurm_args.gpu_type} include an option not in any node")
            
        excluded_nodes = sorted(excluded_nodes + [n for n in node2config if not n in included_nodes])
        all_included_nodes = [n for n in included_nodes if not n in excluded_nodes]
        all_included_nodes = [n for n in all_included_nodes if node2config[n]["gpus_per_node"] >= gpus_per_node]
        if len(all_included_nodes) == 0:
            raise ValueError(f"Could not find a node with {gpus_per_node} GPUs and {slurm_args.mem} RAM")
        
        # Do this last, since we need to request the minimum number of CPUs across all
        # nodes, so we want to filter out as many nodes as possible that could drive
        # this number down first.
        cpus_per_gpu = max(min([node2config[n]["cpus_per_gpu"] for n in all_included_nodes]), args.num_workers)
        if cpus_per_gpu * gpus_per_node > min([node2config[n]["cpus_per_gpu"] * node2config[n]["gpus_per_node"] for n in all_included_nodes]):
            raise ValueError(f"Requested {args.num_workers} per GPU workers but only {min([node2config[n]['cpus_per_gpu'] * node2config[n]['gpus_per_node'] for n in all_included_nodes]):} CPUs available on some otherwise-allocatable nodes")

        mem = min([gpus_per_node * node2config[n]["mem_per_gpu"] for n in all_included_nodes])
        constraint = None
        full_node = all([node2config[n]["gpus_per_node"] == len(args.gpus) for n in all_included_nodes])
        extra_env_vars = slurm_args.extra_env_vars if "extra_env_vars" in slurm_args else []

    elif get_cluster_type() in ["beluga", "cs_apex"]:
        cluster_spec = cluster2node2config[get_cluster_type()]["default"]
        excluded_nodes = [n for n in slurm_args.exclude if n.startswith(cluster2node_prefix[get_cluster_type()])]
        cpus_per_gpu = max(cluster_spec["cpus_per_gpu"], args.num_workers)
        if cpus_per_gpu * gpus_per_node > cluster_spec["cpus_per_gpu"] * cluster_spec["gpus_per_node"]:
            raise ValueError(f"Requested {args.num_workers} per GPU workers but only {cluster_spec['cpus_per_gpu'] * cluster_spec['gpus_per_node']} CPUs available on some otherwise-allocatable nodes")
        mem = cluster_spec["mem_per_gpu"] * gpus_per_node if slurm_args.mem == "adapt" else int(slurm_args.mem.replace("G", ""))
        gpu_type = cluster_spec["gpu_type"] if slurm_args.gpu_type == "adapt" else slurm_args.gpu_type
        constraint = None
        full_node = None

        extra_env_vars = [f"{k}={v}" for k,v in cluster_spec["extra_env_vars"].items()] if "extra_env_vars" in cluster_spec else []
        extra_env_vars = extra_env_vars + slurm_args.extra_env_vars if "extra_env_vars" in slurm_args else extra_env_vars

    elif get_cluster_type() == "narval":
        slurm_args.gpu_type = "a100" if slurm_args.gpu_type == "adapt" else slurm_args.gpu_type
        cluster_spec = cluster2node2config[get_cluster_type()][slurm_args.gpu_type]
        excluded_nodes = [n for n in slurm_args.exclude if n.startswith(cluster2node_prefix[get_cluster_type()])]
        cpus_per_gpu = max(cluster_spec["cpus_per_gpu"], args.num_workers)
        if cpus_per_gpu * gpus_per_node > cluster_spec["cpus_per_gpu"] * cluster_spec["gpus_per_node"]:
            raise ValueError(f"Requested {args.num_workers} per GPU workers but only {cluster_spec['cpus_per_gpu'] * cluster_spec['gpus_per_node']} CPUs available on some otherwise-allocatable nodes")
        mem = cluster_spec["mem_per_gpu"] * gpus_per_node if slurm_args.mem == "adapt" else int(slurm_args.mem.replace("G", ""))
        gpu_type = slurm_args.gpu_type
        constraint = None
        full_node = None

        extra_env_vars = [f"{k}={v}" for k,v in cluster_spec["extra_env_vars"].items()] if "extra_env_vars" in cluster_spec else []
        extra_env_vars = extra_env_vars + slurm_args.extra_env_vars if "extra_env_vars" in slurm_args else extra_env_vars

    elif get_cluster_type() == "cedar":
        slurm_args.gpu_type = "v100l" if slurm_args.gpu_type == "adapt" else slurm_args.gpu_type
        cluster_spec = cluster2node2config[get_cluster_type()][slurm_args.gpu_type]
        excluded_nodes = [n for n in slurm_args.exclude if n.startswith(cluster2node_prefix[get_cluster_type()])]
        cpus_per_gpu = max(cluster_spec["cpus_per_gpu"], args.num_workers)
        if cpus_per_gpu * gpus_per_node > cluster_spec["cpus_per_gpu"] * cluster_spec["gpus_per_node"]:
            raise ValueError(f"Requested {args.num_workers} per GPU workers but only {cluster_spec['cpus_per_gpu'] * cluster_spec['gpus_per_node']} CPUs available on some otherwise-allocatable nodes")
        mem = cluster_spec["mem_per_gpu"] * gpus_per_node if slurm_args.mem == "adapt" else int(slurm_args.mem.replace("G", ""))
        gpu_type = slurm_args.gpu_type
        constraint = None
        full_node = None

        extra_env_vars = [f"{k}={v}" for k,v in cluster_spec["extra_env_vars"].items()] if "extra_env_vars" in cluster_spec else []
        extra_env_vars = extra_env_vars + slurm_args.extra_env_vars if "extra_env_vars" in slurm_args else extra_env_vars

    # See https://slurm.schedmd.com/sbatch.html for setting constraints on SLURM jobs
    elif get_cluster_type() == "graham":
        if slurm_args.gpu_type == "adapt" and UtilsSlurm.time_str_to_hours(slurm_args.time) <= 3:
            slurm_args.gpu_type = "v100,v100l,a5000,a100"
        elif slurm_args.gpu_type == "adapt" and UtilsSlurm.time_str_to_hours(slurm_args.time) > 3:
            slurm_args.gpu_type = "v100,v100l"
        
        # While in other clusters 'v100' means a 16GB V100, here we interpret it to
        # also include 32GB ones. All V100 nodes are annotated with 'v100' regardless
        # of the actual VRAM
        if slurm_args.gpu_type in ["v100", "a5000", "a100"]:
            gpu_type = slurm_args.gpu_type
            constraint = None
            cluster_spec = cluster2node2config["graham"][gpu_type]
            cpus_per_gpu = min(cluster_spec["cpus_per_gpu"], args.num_workers)
            if cpus_per_gpu * gpus_per_node > cluster_spec["cpus_per_gpu"] * cluster_spec["gpus_per_node"]:
                raise ValueError(f"Requested {args.num_workers} per GPU workers but only {cluster_spec['cpus_per_gpu'] * cluster_spec['gpus_per_node']} CPUs available on some otherwise-allocatable nodes")
            mem = cluster_spec["mem_per_gpu"] * gpus_per_node
        else:
            gpu_type = None

            node2config = cluster2node2config["graham"]
            node_types = [n for n in node2config if node2config[n]["gpu_type"] in slurm_args.gpu_type.split(",")]
            constraint = "|".join([f"({node2config[n]['constraint']})" for n in node_types])
            cpus_per_gpu = max(min([node2config[n]["cpus_per_gpu"] for n in node_types]), args.num_workers)
            if cpus_per_gpu * gpus_per_node > min([node2config[n]["cpus_per_gpu"] * node2config[n]["gpus_per_node"] for n in node_types]):
                raise ValueError(f"Requested {args.num_workers} per GPU workers but only {min([node2config[n]['cpus_per_gpu'] * node2config[n]['gpus_per_node'] for n in node_types]):} CPUs available on some otherwise-allocatable nodes")
            mem = min([gpus_per_node * node2config[n]["mem_per_gpu"] for n in node_types]) if slurm_args.mem == "adapt" else int(slurm_args.mem.replace("G", ""))

        full_node = None
        excluded_nodes = [n for n in slurm_args.exclude if n.startswith(cluster2node_prefix[get_cluster_type()])]

        extra_env_vars = slurm_args.extra_env_vars if "extra_env_vars" in slurm_args else []
    else:
        raise NotImplementedError(f"Cluster {get_cluster_type()} not supported")

    
    return dict(cpus_per_task=cpus_per_gpu * (1 if distributed else len(args.gpus)),
        ntasks_per_node=gpus_per_node if distributed else 1,
        cpus_per_gpu=cpus_per_gpu,  nodes=nodes, gpus_per_node=gpus_per_node,
        total_gpus=total_gpus, mem=mem, gpu_type=gpu_type, constraint=constraint,
        exclude=excluded_nodes, full_node=full_node, extra_env_vars=extra_env_vars)
    
def json_to_dict(json_file):
    """Returns a dictionary from a JSON string."""
    import json
    with open(json_file, "r") as json_file:
        return json.load(json_file)

def get_exp_name(*, get_save_folder, total_gpus, time):
    """Returns a job name based on the arguments in [args]. Doesn't need to be parsed,
    for SLURM scripts primarily.

    The arguments [array_start] and [array_end] aren't used as indices directly. The
    task index is included in the job's output and squeue, so it's not needed.
    
    Args:
    get_save_folder    -- model folder
    total_gpus        -- number of GPUs
    time            -- time string of form "HH:MM:SS" or "D-HH:MM:SS"
    array_start     -- array start (if array job)
    array_end       -- array end (if array job)
    """
    get_save_folder = osp.basename(get_save_folder)
    time = UtilsSlurm.pretty_time_str(time)
    return f"{get_save_folder}-gpus{total_gpus}-{time}"
        
def get_random_port(max_port_address=65535, min_port_address=3456):
    """Returns a port between [min_port_address] and [max_port_address]."""
    return random.randint(min_port_address, max_port_address)

def wrap_in_one_task(s, time_limit_bash_var=False, per_line=False, prefix="", per_node=False):
    """Wraps [s] in an srun that runs it as a single task."""
    s = s.strip("\n")
    if len(s) == 0:
        return s
    else:
        time_str = f" --time ${time_limit_bash_var} " if time_limit_bash_var else ""
        Nn_str = "-N $SLURM_NNODES -n $SLURM_NNODES" if per_node else "-N 1 -n 1"
        if per_line:
            newline = "\n"
            split_str = [cmd.strip() for l in s.split("\n") for cmd in l.split(";")]
            return f"{newline}{prefix}".join([f"srun {time_str} {Nn_str} {l}" for l in split_str])
        else:
            return f"\nsrun {time_str} {Nn_str} bash << EOF \n{s}\nEOF\nwait\n"

def wrap_in_one_task_per_node(s, time_limit_bash_var=False, prefix="", per_line=False):
    """Wraps [s] in an srun that runs it as a single task per node."""
    return wrap_in_one_task(s, time_limit_bash_var=time_limit_bash_var, prefix=prefix, per_line=per_line, per_node=True)

def get_tar_code_cmd(get_save_folder):
    """Returns a string that tars the code to 'code.tar' inside [get_save_folder].

    Only one process should do this! It should run only when the job starts for the
    first time.
    """
    things_to_tar = ["*.py"]
    
    # On Solar, storage space is a big issue, so we can't tar the .git folder. We
    # instead write a file containing the current git commit. Since the files run will
    # use the version whenever the job starts for the first time,
    things_to_tar += ["cur_git_sha.txt"] if is_solar() else [".git"]
    update_sha_cmd = f"cur_git_sha=$(python UtilsSlurm.py --function get_git_sha)\n\techo $cur_git_sha > cur_git_sha.txt" if is_solar() else "# Not creating cur_git_sha.txt file"

    s = [f"mkdir -p {get_save_folder}", update_sha_cmd]
    for t in sorted(set(things_to_tar)):
        s += [f"tar -rf {get_save_folder}/code.tar {t}"]

    return "\n\t".join(s)

def get_tar_code_to_compute_node_str(get_save_folder):
    """Returns a command that moves code from the code.tar in the model folder to a
    compute node.
    """
    return wrap_in_one_task_per_node(f"tar -xf {get_save_folder}/code.tar -C $SLURM_TMPDIR", per_line=False)

def get_full_python_env_str(*, slurm_args, distributed=False):
    """Returns a Python installation string for the experiment.

    Following https://docs.alliancecan.ca/wiki/Python#Example_(multi-nodes), the
    paradigm seems to by each node creates the environment, and the main task only
    activates it again. Quote: "srun exports the current env, which contains
    $VIRTUAL_ENV and $PATH variables."

    Args:
    slurm_args  -- slurm submission arguments
    distributed -- whether the run will use DDP or not
    """
    if is_solar() and slurm_args.env == "adapt":
        pip_file = None
        conda_unpack_file = "py311OneDDPTutorial.tar.gz"
    elif is_cc() and slurm_args.env == "adapt":
        pip_file = "pip_reqs_cc.txt"
        conda_unpack_file = None
        no_download, no_index = "--no-download", "--no-index"
    elif osp.exists(slurm_args.env) and slurm_args.env.endswith(".txt"):
        pip_file = slurm_args.env
        conda_unpack_file = None
        no_download, no_index = ("--no-download", "--no-index") if is_cc() else ("", "")
    elif osp.exists(slurm_args.env) and slurm_args.env.endswith(".tar") or slurm_args.env.endswith(".tar.gz"):
        pip_file = None
        conda_unpack_file = slurm_args.env
    else:
        pip_file = None
        conda_unpack_file = None

    if not pip_file is None:
        install_str = f"virtualenv {no_download} $SLURM_TMPDIR/py311OneDDPTutorial; source $SLURM_TMPDIR/py311OneDDPTutorial/bin/activate; pip install --no-index --upgrade pip; pip install {no_index} -r {pip_file}\n"
        install_str = wrap_in_one_task_per_node(install_str, per_line=False) if distributed else install_str
        install_str += "source $SLURM_TMPDIR/py311OneDDPTutorial/bin/activate\n" if distributed else ""
        return install_str
    elif not conda_unpack_file is None:
        cp_parent = f"/localscratch/{os.environ.get('USER')}" if is_solar() and slurm_args.full_node else "$SLURM_TMPDIR"
        extract_str = "xzf" if ".gz" or ".tgz" in conda_unpack_file else "xf"
        move_file_str = f"python UtilsSlurm.py --function copy --cp_src {osp.realpath(conda_unpack_file)} --cp_parent {cp_parent} --untar_if_tar 0 --cp_force {1 if slurm_args.cp_force else -1}"
        install_str = f"{move_file_str}\nmkdir -p {cp_parent}/py311OneDDPTutorial\ntar -{extract_str} {cp_parent}/{conda_unpack_file} -C {cp_parent}/py311OneDDPTutorial\n$SLURM_TMPDIR/py311OneDDPTutorial/bin/python\nsource {cp_parent}/py311OneDDPTutorial/bin/activate\n\n"
        return install_str
    else:
        raise NotImplementedError()        

def filter_args_from_str(args_list, args, keep_args=["uid", "wandb"], keep_if_not_none=["resume"]):
    """Returns argparse Namespace [args] such that it contains only ones matching
    unparsed ones in [args_list] or that occur in [keep_args].

    Note: this function isn't entirely robust; if an argument value is the name
    of another unspecified argument, that unspecified argument won't be filtered.
    """
    keep_args += [a for a in keep_if_not_none if a in vars(args) and not vars(args)[a] in [[], None, ""]]
    possible_args = {a.replace("--", "") for a in args_list if a.startswith("--")} | set(keep_args)    
    keep_args = [a for a in sorted(vars(args)) if a in possible_args]
    return argparse.Namespace(**{k: vars(args)[k] for k in keep_args})

def unparse_args(args):
    """Returns [args] as a string that can be parsed again."""
    s = ""
    for k,v in vars(args).items():
        if isinstance(v, (list, tuple)):
            s += f" --{k} {' '.join([str(v_) for v_ in v])}"
        elif v is None:
            continue
        else:
            s += f" --{k} {v}"
    return s

def get_account_str(slurm_args):
    """Returns the account or partition to use for the job."""
    if slurm_args.account == "adapt":
        slurm_args.account = cluster2misc_reqs[get_cluster_type()]["default_account"]

    if is_solar() and slurm_args.account in ["debug", "cs-gpu-research"]:
        return slurm_args.account
    elif is_solar() and not (slurm_args.account.endswith("-short") or slurm_args.account.endswith("-long")):
        short_long = "short" if UtilsSlurm.time_str_to_minutes(slurm_args.time) < 2880 else "long"
        return f"{slurm_args.account}-{short_long}"
    else:
        return slurm_args.account

def set_omni_seed_from_file(slurm_args, args):
    """Returns [args] with [omni_seed] set to the value in the file pointed to by
    args.[set_omni_seed_from].
    """
    if slurm_args.set_omni_seed_from is None:
        return args
    elif not args.omni_seed is None:
        raise ValueError("Cannot set omni_seed from file as it is already set to integer")
    elif not slurm_args.set_omni_seed_from in args:
        raise ValueError(f"Cannot set omni_seed from {slurm_args.set_omni_seed_from} as it is not in args")
    elif not osp.exists(vars(args)[slurm_args.set_omni_seed_from]):
        raise FileNotFoundError(f"Cannot set omni_seed from {slurm_args.set_omni_seed_from} as it does not exist")
    else:
        load_from = f"{osp.dirname(vars(args)[slurm_args.set_omni_seed_from])}/config.json"
        if osp.exists(load_from):
            loaded_args = json_to_dict(load_from)
            if "omni_seed" in loaded_args and not loaded_args["omni_seed"] is None:
                return argparse.Namespace(**vars(args) | dict(omni_seed=loaded_args["omni_seed"]))
            else:
                raise ValueError(f"Cannot set omni_seed from {slurm_args.set_omni_seed_from} as it does not contain 'omni_seed' or it is None")
        else:
            raise FileNotFoundError(f"Cannot set omni_seed from {slurm_args.set_omni_seed_from} as it {load_from} does not exist")


def get_setup_node_str(*, slurm_args, get_save_folder):
    """Returns a string that sets up a compute node. It must
    1. Ensure $SLURM_TMPDIR is reasonable
    2. Ensure a Python installation exists
    3. Ensure the ~/scratch and /localscratch/USER/data folders exist (Solar only)
    4. Ensure the WANB_API_KEY environment variable is correct
    """
    if is_solar():
        user = os.environ.get("USER")
        slurm_tmpdir = f"{UtilsSlurm.get_slurm_tmpdir_folder()}/{osp.basename(get_save_folder)}"
        set_slurm_tmpdir = f"export SLURM_TMPDIR={slurm_tmpdir}\n"
        set_slurm_tmpdir += wrap_in_one_task_per_node("mkdir -p $SLURM_TMPDIR ; rm -rf $SLURM_TMPDIR/*")
        python_setup_str = "# Basic Python installation assumed to exist already"
        scratch_str = wrap_in_one_task_per_node(f"ln -s /project/apex-lab/$USER/scratch ~/scratch")
        data_str = wrap_in_one_task_per_node(f"mkdir -p /localscratch/$USER/data ; ln -s /localscratch/$USER/data $SLURM_TMPDIR/data")
        # Only if you actually use WandB
        # wandb_str = f"export WANDB_API_KEY=$(cat /project/apex-lab/$USER/.wandb_api_key.txt)\n"
        wandb_str = ""
    elif is_cc():
        set_slurm_tmpdir = "# $SLURM_TMPDIR assumed to exist already"
        python_setup_str = f"module load python/3.11"
        scratch_str = ""
        data_str = ""
        wandb_str = ""
    elif get_cluster_type() == "cs_apex":
        set_slurm_tmpdir = "# $SLURM_TMPDIR assumed to exist already"
        python_setup_str = f"source ~/py311OneDDPTutorial/bin/activate"
        scratch_str = ""
        data_str = ""
        wandb_str = ""
    else:
        raise NotImplementedError()
    return "\n".join([set_slurm_tmpdir, python_setup_str, scratch_str, data_str, wandb_str])

def get_env_vars_str(slurm_args):
    """Writes environment variabels specified in [slurm_args] as things exported
    inside the SLURM script along with some defaults to make DDP work.
    """
    default = dict(
        PYTHONUNBUFFERED="1",
        MASTER_PORT=get_random_port(),
        MASTER_ADDR="$(hostname)",
        TRITON_HOME="$SLURM_TMPDIR",
        TORCH_NCCL_BLOCKING_WAIT="1",
        OMP_NUM_THREADS="1", # Maybe fixes Imagenet problems on Beluga?
        TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1,
    )
    extra = [kv.split("=") for kv in slurm_args.extra_env_vars]
    extra = {k: v for k,v in extra}
    if not len(extra) == 0:
        tqdm.write(f"Extra environment variables: {extra}")
    env_vars = default | extra
    return "\n".join(f"export {k}={v}" for k,v in env_vars.items())

def get_comment_str(*, slurm_args, args):
    """Returns the string to comment each SLURM job with. Ideally should be run after
    [args] is modified with filter_args_from_str().
    """
    # Exluding arguments prefixed with 'wandb_' prevents an "sbatch: error: Batch
    # job submission failed: Pathname of a file, directory or other parameter too
    # long" error on Solar. I think it might be referring to the comment.
    comment = {k: v for k,v in vars(args).items() if not k.startswith("wandb_")}
    comment = json.dumps(comment)
    comment = comment.replace("\"", "'")
    return "\"" + comment + "\""
        
def get_slurm_args():

    def gpu_type_is_valid(g):
        """Returns if a gpu type string is valid."""
        if all([t for t in g.split(",") in SlurmSubmit.gpu2vram.keys() | {"adapt", "big", "any"}]):
            return g
        else:
            raise ValueError(f"GPU spec={g} invalid")


    P = argparse.ArgumentParser(allow_abbrev=False)
    P.add_argument("script",
        help="Script to run")
    
    # SLURM config
    P.add_argument("--time", required=True, type=str,
        help="String giving time for the SLURM job")
    P.add_argument("--account", default="adapt", choices=["adapt", "def-keli", "rrg-keli", "cs-gpu-research", "apex-lab", "debug"],
        help="String giving the account for the SLURM job, or on Solar, the partition")
    P.add_argument("--user_email", default="tristanengst@gmail.com", type=str,
        help="String giving time for the SLURM job")
    
    # Hardware config
    P.add_argument("--gpu_type", default="adapt",
        help="Comma-separated string of GPU types, or 'adapt' to select automatically")
    P.add_argument("--mem", default="adapt", type=str,
        help="RAM—specify SLURM argument like '100G', or 'adapt' to select automatically")
    P.add_argument("--exclude", default=[], nargs="+",
        help="Exclude these nodes if on a matching cluster")

    # Software config
    P.add_argument("--env", default="adapt",
        help="Python environment type. 'adapt' is the cluster default. Existing .tar.gz files are conda-unpacked. .txt files are pip installed. Anything else is interpreted as the entire installation string")
    P.add_argument("--dpp_launch_method", default="python", choices=["python", "torchrun"],
        help="DDP launch method. 'python' works fine on CC and Solar with more than one task. 'torchrun' is the alternative.")

    # Logging
    P.add_argument("--disable_wandb", choices=[0, 1], default=0, type=int,
        help="Force wandb to 'disabled")
    P.add_argument("--job_results_dir", default=None,
        help="Directory to save job results to if different from the one assumed from SCRIPT")
    P.add_argument("--job_results_use_suffix", choices=[0, 1], default=0, type=int,
        help="Use DEFAULT_JOB_RESULTS_DIR/SUFFIX as the job results directory")

    # Resuming/using existing experiments
    P.add_argument("--set_resume_with_uid", choices=[0, 1], default=0, type=int,
        help="Set RESUME intelligently")
    P.add_argument("--resubmit_epi_time", type=int, default=None,
        help="Amount of time needed to resubmit job at end of job. Set to zero to turn off resubmission. None will be changed to 5 IFF 'resume' is in the script args and set to 'latest'")
    P.add_argument("--set_omni_seed_from", default=None,
        help="Sets [omni_seed] from the given file argument")
    P.add_argument("--cp_force", default=0, choices=[0, 1], type=int,
        help="Forcibly move data files even if they're already on the node")

    # Miscellaneous
    P.add_argument("--dry_run", choices=[0, 1], default=0, type=int,
        help="0: submit via sbatch, 1: print, no submit")
    P.add_argument("--extra_env_vars", default=[], nargs="*",
        help="Space-separated ENV_VAR=X string. These are written to additional ones with defaults important for DDP.")

    # Args that need to be copied into [unparsed_args]
    P.add_argument("--data_allow_sf", choices=[0, 1], default=1, type=int,
        help="Allow upgrading to safetensors data if available")

    return P.parse_known_args()

if __name__ == "__main__":
    slurm_args, unparsed_args = get_slurm_args()

    # Each script must set the following variables:
    # job_results_str               -- file where outputs will be printed to
    # distributed                   -- whether or DDP will be used
    # file_move_command             -- command that moves data to the compute node
    # checkpoints_saved_to_folder   -- folder where things are saved
    # exp_name                      -- checkpoints should be saved under checkpoints_saved_to_folder/exp_name
    # args                          -- argparse for the experiment that will be run;
    #                                   it may get modified later in this script, and
    #                                   is expected to contain important information
    if slurm_args.script in ["TrainAndEval.py",]:

        if slurm_args.script == "TrainAndEval.py":
            from TrainAndEval import get_args as get_args
            from TrainAndEval import get_save_folder as get_save_folder
            job_results_str = "pretrain_results"

        args = get_args(unparsed_args, on_compute_node=False)

        # Lots of things work differently with DDP enabled. We assume only one node
        distributed = "ddp" in args.speedup

        # Set the job results directory
        if slurm_args.job_results_use_suffix and not slurm_args.job_results_dir is None:
            raise ValueError("Cannot set both SUFFIX_AS_RESULTS_DIR and JOB_RESULTS_DIR")
        elif "suffix" in args and slurm_args.job_results_use_suffix and slurm_args.job_results_dir is None:
            job_results_str = f"{job_results_str}/{args.suffix}".replace(".", "_")
        elif not "suffix" in args and slurm_args.job_results_use_suffix and slurm_args.job_results_dir is None:
            raise ValueError("SUFFIX_AS_RESULTS_DIR set but no suffix in args")
        elif not slurm_args.job_results_dir is None:
            job_results_str = slurm_args.job_results_dir
        else:
            job_results_str = job_results_str

        ##############################################################################
        # Generate a command that moves all files onto the compute node appropriately.
        # When running on ComputeCanada, this should just dump the relevant files onto
        # the compute node as needed. On Solar, $SLURM_TMPDIR/data is symlinked to
        # ~/data for each node, so any file prefixed with data/... will stay on the
        # compute node permanently, and not need to be copied again.
        ##############################################################################
        file_move_command = ""
        if args.data_tr == "cifar10" or args.data_val == "cifar10":
            src, parent = "cifar-10-python.tar.gz", f"$SLURM_TMPDIR/"
            file_move_command += f"python UtilsSlurm.py --function copy --cp_src {src} --cp_parent {parent} --cp_force {slurm_args.cp_force} --untar_if_tar 0\n"
            src, parent = "cifar-10-batches-py", f"$SLURM_TMPDIR/"
            file_move_command += f"python UtilsSlurm.py --function copy --cp_src {src} --cp_parent {parent} --cp_force {slurm_args.cp_force} --untar_if_tar 0\n"
        if args.data_tr == "cifar100" or args.data_val == "cifar100":
            src, parent = "cifar-100-python.tar.gz", f"$SLURM_TMPDIR/"
            file_move_command += f"python UtilsSlurm.py --function copy --cp_src {src} --cp_parent {parent} --cp_force {slurm_args.cp_force} --untar_if_tar 0\n"
            src, parent = "cifar-100-python", f"$SLURM_TMPDIR/"
            file_move_command += f"python UtilsSlurm.py --function copy --cp_src {src} --cp_parent {parent} --cp_force {slurm_args.cp_force} --untar_if_tar 0\n"
        if args.data_tr == "mnist" or args.data_val == "mnist":
            src, parent = "MNIST", f"$SLURM_TMPDIR/"
            file_move_command += f"python UtilsSlurm.py --function copy --cp_src {src} --cp_parent {parent} --cp_force {slurm_args.cp_force} --untar_if_tar 0\n"

        # REMOVED SINCE THE TUTORIAL IGNORES SAFETENSORS DATA
        # If safetensors files are available, we should switch datasets to use them
        # unless [args] has [data_allow_sf] set to False. 
        # args = args_with_sf_data_set(args, keys=["data_tr", "data_val"])

        if osp.exists(args.data_tr) and not (is_solar() and args.data_tr in ["cifar10", "cifar100", "mnist"]):
            src = osp.realpath(args.data_tr)
            parent = f"$SLURM_TMPDIR/{osp.relpath(osp.dirname(args.data_tr))}"
            file_move_command += f"python UtilsSlurm.py --function copy --cp_src {src} --cp_parent {parent} --cp_force {slurm_args.cp_force} --untar_if_tar 1 \n"
            args.data_tr = args.data_tr.rstrip(".tar")
        elif is_solar() and args.data_tr in ["cifar10", "cifar100", "mnist"]:
            pass
        else:
            raise FileNotFoundError(args.data_tr)
        
        if osp.exists(args.data_val) and not (is_solar() and args.data_val in ["cifar10", "cifar100", "mnist"]):
            src = osp.realpath(args.data_val)
            parent = f"$SLURM_TMPDIR/{osp.relpath(osp.dirname(args.data_val))}"
            file_move_command += f"python UtilsSlurm.py --function copy --cp_src {src} --cp_parent {parent} --cp_force {slurm_args.cp_force} --untar_if_tar 1 \n"
            args.data_val = args.data_val.rstrip(".tar")
        elif is_solar() and args.data_tr in ["cifar10", "cifar100", "mnist"]:
            pass
        else:
            raise FileNotFoundError(args.data_val)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################

        gpus_per_node = len(args.gpus)
        total_gpus = gpus_per_node * args.nodes
        checkpoints_saved_to_folder = get_save_folder(args)
        exp_name = get_exp_name(get_save_folder=checkpoints_saved_to_folder,
            total_gpus=total_gpus,
            time=slurm_args.time)

        ##############################################################################
        # Usually I have the resume arguments set to --latest rather than a specific
        # checkpoint, and there's a function that finds the file to resume from.
        ##############################################################################
        # If --resubmit_epi_time is None, it might still be set to a value
        if "resume" in args and args.resume == "latest" and slurm_args.resubmit_epi_time is None:
            slurm_args.resubmit_epi_time = 2
        elif slurm_args.resubmit_epi_time is None:
            slurm_args.resubmit_epi_time = 0
        else:
            pass

        # For each argument ending in 'save_iter_t' that is not zero, change it to the
        # maximum of 5 minutes, and the minimum of itself and the time requested for
        # the job divided by 16. This is helpful for jobs that can be preempted, as
        # otherwise they can lose a lot of progress.
        for k,v in vars(args).items():
            if k.endswith("save_iter_t") and not v == 0:
                vars(args)[k] = int(max(5, min(v, UtilsSlurm.time_str_to_minutes(slurm_args.time) / 16)))

    else:
        job_results_str = None
        distributed = None
        file_move_command = None
        checkpoints_saved_to_folder = None
        exp_name = None
        args = None
        raise NotImplementedError()

    # Imporant check as this is otherwise difficult to debug
    if slurm_args.resubmit_epi_time >= UtilsSlurm.time_str_to_minutes(slurm_args.time) - 1:
        tqdm.write(f"\n\n\n WARNING: epilogue time too big relative to requested time. This wil usually cause issues!\n\n\n")

    
    ##################################################################################
    # Host specific settings
    ##################################################################################
    node_reqs_dict = get_node_reqs(args=args, slurm_args=slurm_args, distributed=distributed)
    slurm_args = argparse.Namespace(**vars(slurm_args) | node_reqs_dict)
    args.wandb = "disabled" if slurm_args.disable_wandb else cluster2misc_reqs[get_cluster_type()]["wandb_default_mode"]
    slurm_args.account = get_account_str(slurm_args)
    
    ##############################################################################
    # This is important only if you're actually using WandB
    ##############################################################################
    # Set defaults for WandB unless they're in the unparsed args
    # args.wandb_dir = osp.expanduser("~/scratch/IMLE-SSL") if args.wandb_dir is None else args.wandb_dir
    # args.wandb_cache_dir = osp.expanduser("~/scratch/IMLE-SSL/wandb_cache") if args.wandb_cache_dir is None else args.wandb_cache_dir
    # args.wandb_data_dir = osp.expanduser("~/scratch/IMLE-SSL/wandb_staging") if args.wandb_data_dir is None else args.wandb_data_dir

    args = filter_args_from_str(unparsed_args, args, keep_if_not_none=["resume", "wandb_dir", "wandb_cache_dir", "wandb_data_dir", "omni_seed"] + [k for k in vars(args) if k.endswith("save_iter_t")])

    # Set the launch method appropriately. Impacts both DDP and non-DDP. In general it should be 'python' and on clusters DDP enabled with --ntasks set to more than one
    launch_command = "python"

    tar_code_str = get_tar_code_cmd(checkpoints_saved_to_folder)                        # tar code in the submitted-from folder to the model folder (first task only)
    file_move_command = wrap_in_one_task_per_node(file_move_command)                    # Move data to the compute node
    
    
    # TODO: On ComputeCanada one node, we cannot wrap this in srun as it isn't executed
    # and breaks. On Solar multinode, it was necessary.... this is an issue?
    move_on_node_str = "cd $SLURM_TMPDIR"                                               # Just a plain CD
    # move_on_node_str = wrap_in_one_task_per_node(move_on_node_str) if slurm_args.nodes > 1 and is_solar() else move_on_node_str

    # After moving to the compute node, install the Python environment that can run the experiment. Python should be made available before this
    second_python_str = get_full_python_env_str(slurm_args=slurm_args, distributed=distributed)

    # Set the time limit for the experiment so that there is just enough to resubmit it if it doesn't finish
    timelimit_cmd = f"timelimit=$(python UtilsSlurm.py --function get_time_limit --epi_time {slurm_args.resubmit_epi_time})"
    
    # Hack that might work from https://stackoverflow.com/questions/77255279/slurm-srun-with-multiple-executables.
    # Note that we put the ending quote at the end of [script].
    # This includes the timelimit in srun, so it will not run over and prevent resubmission
    submit_command = f"srun --nodes {slurm_args.nodes} --ntasks-per-node {slurm_args.ntasks_per_node} --cpus-per-task {slurm_args.cpus_per_task} --time $timelimit bash -c ' {launch_command} {slurm_args.script} {unparse_args(args)} "
    script = f"{file_move_command}\n\n{move_on_node_str}\n\n{second_python_str}\n\n{timelimit_cmd}\n{submit_command} "
    script += f" --job_id $SLURM_JOB_ID --num_workers {slurm_args.cpus_per_gpu} --compute_node $SLURM_TMPDIR --save_folder ~/scratch/IMLE-SSL --submit_dir {osp.abspath(osp.dirname(__file__))}'"
    slurm_template = "slurm/slurm_template.txt"

    with open(slurm_template, "r") as f:
        slurm_template = f.read()

    slurm_script = f"slurm/{exp_name}.sh"
    slurm_template = slurm_template.replace("TIME", slurm_args.time)
    slurm_template = slurm_template.replace("ACCOUNT_OR_PARTITION", f"--{'partition' if is_solar() else 'account'}={slurm_args.account}")
    slurm_template = slurm_template.replace("NUM_NODES", f"{slurm_args.nodes}")
    slurm_template = slurm_template.replace("GPU_STR", f"{slurm_args.gpus_per_node}" if slurm_args.gpu_type is None else f"{slurm_args.gpu_type}:{slurm_args.gpus_per_node}")
    slurm_template = slurm_template.replace("TASKS_PER_NODE", f"{slurm_args.ntasks_per_node}")
    slurm_template = slurm_template.replace("MEM_PER_NODE", f"{slurm_args.mem}G")
    slurm_template = slurm_template.replace("CPUS_PER_TASK", f"{slurm_args.cpus_per_task}")
    slurm_template = slurm_template.replace("EXCLUDED_NODES", ",".join(slurm_args.exclude))
    slurm_template = slurm_template.replace("CONSTRAINT", "" if slurm_args.constraint is None else slurm_args.constraint)
    slurm_template = slurm_template.replace("COMMENT", get_comment_str(slurm_args=slurm_args, args=args))
    
    slurm_template = slurm_template.replace("JOB_RESULTS_STR", job_results_str)
    # These next two must be done in order!
    slurm_template = slurm_template.replace("JOB_PREFIX", "preempt_me_" if is_solar() and slurm_args.account in ["debug", "cs-gpu-research"] else "")
    slurm_template = slurm_template.replace("NAME", f"{exp_name}")
    
    slurm_template = slurm_template.replace("USER_EMAIL", str(slurm_args.user_email))

    # Set up the node(s) and DDP
    slurm_template = slurm_template.replace("NODE_SETUP_STR", get_setup_node_str(
        slurm_args=slurm_args,
        get_save_folder=checkpoints_saved_to_folder))
    # slurm_template = slurm_template.replace("COMPUTED_MASTER_PORT", f"{get_random_port()}")

    slurm_template = slurm_template.replace("MOVE_CODE_TO_COMPUTE_NODE_STR", get_tar_code_to_compute_node_str(checkpoints_saved_to_folder))

    slurm_template = slurm_template.replace("ENV_VARS_STR", get_env_vars_str(slurm_args))
   
    # Finally add the script
    slurm_template = slurm_template.replace("SCRIPT", script)

    slurm_template = slurm_template.replace("get_save_folder", checkpoints_saved_to_folder)
    slurm_template = slurm_template.replace("TAR_CODE_CMD", tar_code_str)
    slurm_template = slurm_template.replace("SBATCHSH", f"{osp.dirname(__file__)}/{slurm_script}")
    slurm_template = slurm_template.replace("EPI", str(slurm_args.resubmit_epi_time))
    slurm_template = slurm_template.replace("SUBMITED_FROM_DIR", osp.abspath(osp.dirname(__file__)))

    ##################################################################################
    ##################################################################################
    ##################################################################################

    
    tqdm.write(f"SLURM submission script written to {slurm_script}")
    tqdm.write(f"Outputs will write to {job_results_str}/{exp_name}.txt")

    if slurm_args.dry_run == 0:
        with open(slurm_script, "w+") as f:        
            f.write(slurm_template)
        os.system(f"sbatch {slurm_script}")
        _ = os.makedirs(f"{osp.abspath(osp.dirname(__file__))}/{job_results_str}", exist_ok=True)
    elif slurm_args.dry_run == 1:
        tqdm.write(f"\n\n\n\n============ PRINTING SLURM TEMPLATE ============")
        tqdm.write(slurm_template)
    else:
        raise NotImplementedError()















       


