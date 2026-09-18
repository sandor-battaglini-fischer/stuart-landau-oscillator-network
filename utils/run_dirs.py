import glob
import json
import os
import tempfile
import uuid

import torch


def task_results_dir(project_root, task):
    return os.path.join(project_root, 'results', task)


def make_run_dir(project_root, task, timestamp, sweep_idx=None, sweep_suffix=None):
    base = task_results_dir(project_root, task)
    os.makedirs(base, exist_ok=True)
    if sweep_idx is not None:
        if sweep_suffix:
            name = f'{timestamp}_sweep{sweep_idx:03d}_{sweep_suffix}'
        else:
            name = f'{timestamp}_sweep{sweep_idx:03d}'
    else:
        name = timestamp

    # Parallel grid workers often share the same second-resolution timestamp.
    # Always uniquify so concurrent runs never share an output directory.
    unique = f'{name}_p{os.getpid()}_{uuid.uuid4().hex[:8]}'
    run_dir = os.path.join(base, unique)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def atomic_write_json(path, payload, indent=2):
    path = str(path)
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix='.json', dir=directory)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def sweep_summary_path(project_root, task, sweep_type, sweep_timestamp):
    base = task_results_dir(project_root, task)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f'sweep_{sweep_type}_{sweep_timestamp}_summary.txt')


def epoch_dir(run_dir, epoch):
    path = os.path.join(run_dir, 'epochs', f'epoch{epoch:02d}')
    os.makedirs(path, exist_ok=True)
    return path


def save_training_checkpoint(model, run_dir, is_best=False):
    last_path = os.path.join(run_dir, 'last_model.pt')
    torch.save(model.state_dict(), last_path)
    if is_best:
        torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))
    for old_checkpoint in glob.glob(os.path.join(run_dir, 'epoch*.pt')):
        os.remove(old_checkpoint)


def create_gif_from_epoch_dirs(run_dir, pattern, output_filename, target_size=None):
    from utils.plotting_utils import create_gif_from_files

    files = sorted(glob.glob(os.path.join(run_dir, 'epochs', 'epoch*', pattern)))
    if not files:
        return
    create_gif_from_files(run_dir, pattern, output_filename, target_size=target_size, files=files)


def promote_epoch_artifacts(ep_dir, run_dir, filename_map):
    import shutil

    for src_name, dst_name in filename_map.items():
        src = os.path.join(ep_dir, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dir, dst_name))
