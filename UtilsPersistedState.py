"""Utilities for persisting state across a SLURM job's lifetime. Like environment
variables, backed by a file and thus with nicer semantics.
"""
import os
import os.path as osp
import json

def get_persisted_state_file():
    """Returns the path to the persisted state file."""
    return f"{os.environ['SLURM_TMPDIR'] if 'SLURM_JOB_ID' in os.environ else '.'}/persisted_state.json"

def dict_to_json(d, f):
    """Saves dictionary [d] as JSON file [f]."""
    _ = os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w+") as f:
        json.dump(d, f, indent=4)

def json_to_dict(f):
    """Returns JSON file [f] as a dictionary."""
    with open(f, "r") as f:
        return json.load(f)

def dict_append_json(d, f):
    """Appends dictionary [d] to JSON file [f]."""
    extant = json_to_dict(f) if osp.exists(f) else dict()
    _ = dict_to_json(extant | d, f)

def persisted_state_update(**kwargs):
    """Sets key [k] to value [v] in persisted state."""
    if not osp.exists(get_persisted_state_file()):
        _ = dict_to_json(kwargs, get_persisted_state_file())
    else:
        _ = dict_append_json(kwargs, get_persisted_state_file())

def persisted_state_get(k, default=None):
    """Returns the value of [k] in persisted state or [default] if it isn't found."""
    if not osp.exists(get_persisted_state_file()):
        return default() if callable(default) else default
    else:
        state = json_to_dict(get_persisted_state_file())
        return state[k] if k in state else (default() if callable(default) else default)

def persistent_state_del(k):
    """Deletes key [k] from the persisted state."""
    if osp.exists(get_persisted_state_file()):
        state = json_to_dict(get_persisted_state_file())
        state = {k_: v for k_,v in state.items() if not k_ == k}
        _ = dict_to_json(state, get_persisted_state_file())

def persisted_state_clear():
    """Removes the persisted state file."""
    os.remove(get_persisted_state_file()) if osp.exists(get_persisted_state_file()) else None

def persisted_state_get_all():
    return json_to_dict(get_persisted_state_file())
