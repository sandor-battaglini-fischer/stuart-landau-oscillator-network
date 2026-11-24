import glob
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime

import torch


def make_run_dir(project_root, task, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    base = os.path.join(project_root, 'results', task)
    os.makedirs(base, exist_ok=True)
    # parallel grid workers can share a timestamp, so add pid + random suffix
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base, f'{timestamp}_p{os.getpid()}_{uuid.uuid4().hex[:8]}')
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def write_log_header(fh, params):
    fh.write('=' * 60 + '\n')
    fh.write('Training Parameters\n')
    fh.write('=' * 60 + '\n')
    for key, value in params.items():
        fh.write(f'{key}: {value}\n')
    fh.write('=' * 60 + '\n')
    fh.flush()


def append_metrics(path, record):
    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    history.append(record)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


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


def epoch_dir(run_dir, epoch):
    path = os.path.join(run_dir, 'epochs', f'epoch{epoch:02d}')
    os.makedirs(path, exist_ok=True)
    return path


def save_initial_checkpoint(model, run_dir):
    torch.save(model.state_dict(), os.path.join(run_dir, 'initial_model.pt'))


def save_training_checkpoint(model, run_dir, is_best=False):
    torch.save(model.state_dict(), os.path.join(run_dir, 'last_model.pt'))
    if is_best:
        torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))
    for old_checkpoint in glob.glob(os.path.join(run_dir, 'epoch*.pt')):
        os.remove(old_checkpoint)


def create_gif_from_epoch_dirs(run_dir, pattern, output_filename, target_size=None):
    from utils.plotting_utils import create_gif_from_files

    files = sorted(glob.glob(os.path.join(run_dir, 'epochs', 'epoch*', pattern)))
    if files:
        create_gif_from_files(run_dir, pattern, output_filename, target_size=target_size, files=files)


def promote_epoch_artifacts(ep_dir, run_dir, filename_map):
    for src_name, dst_name in filename_map.items():
        src = os.path.join(ep_dir, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dir, dst_name))
