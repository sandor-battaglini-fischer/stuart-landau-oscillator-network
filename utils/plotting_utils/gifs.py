import glob
import os

import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def create_gif_from_files(output_dir, pattern, output_filename, target_size=None, files=None):
    if not HAS_IMAGEIO and not HAS_PIL:
        return

    if files is None:
        files = sorted(glob.glob(os.path.join(output_dir, pattern)))
    if not files:
        return

    images = []
    for file in files:
        try:
            if HAS_IMAGEIO:
                img = imageio.imread(file)
            else:
                img = Image.open(file)
                if target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                img = np.array(img)
            images.append(img)
        except Exception:
            continue

    if not images:
        return

    output_path = os.path.join(output_dir, output_filename)
    try:
        if HAS_IMAGEIO:
            imageio.mimsave(output_path, images, duration=0.5)
        else:
            images_pil = [Image.fromarray(img) for img in images]
            if target_size:
                images_pil = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images_pil]
            images_pil[0].save(output_path, save_all=True, append_images=images_pil[1:], duration=500, loop=0)
    except Exception:
        pass
