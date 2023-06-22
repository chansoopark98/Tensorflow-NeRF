import tensorflow as tf
import os
import json
from PIL import Image
import numpy as np
import re
from typing import Tuple
from tqdm import tqdm

def create_meshgrid(height: int, width: int, normalized_coordinates: bool = True) -> tf.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the PyTorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]`.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.
    """
    xs = tf.linspace(tf.cast(0.0, tf.float32), tf.cast(width - 1, tf.float32), width)
    ys = tf.linspace(tf.cast(0.0, tf.float32), tf.cast(height - 1, tf.float32), height)
    
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2

    # generate grid by stacking coordinates
    xs, ys = tf.meshgrid(xs, ys, indexing="xy")
    base_grid = tf.stack([xs, ys], axis=-1)
    return tf.expand_dims(base_grid, axis=0)  # 1xHxWx2

def get_rays(directions: tf.Tensor, c2w: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = tf.linalg.matmul(directions, tf.transpose(c2w[:3, :3]))  # (H, W, 3)
    # rays_d = rays_d / tf.norm(rays_d, axis=-1, keepdims=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = tf.broadcast_to(c2w[:3, 3], tf.shape(rays_d))  # (H, W, 3)

    rays_d = tf.reshape(rays_d, [-1, 3])
    rays_o = tf.reshape(rays_o, [-1, 3])

    return rays_o, rays_d

def get_ray_directions(H: int, W: int, focal: list, center=None) -> tf.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = tf.unstack(grid, axis=-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = tf.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], tf.ones_like(i)], -1)  # (H, W, 3)
    return directions

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


class BlenderDatasetTF:
    def __init__(self, datadir: str, split: str = 'train', downsample: int = 1.0, is_stack: bool = False, N_vis: int=-1):
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        self.scene_bbox = tf.constant([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0,6.0]
        
        self.center = tf.reduce_mean(self.scene_bbox, axis=0).numpy().reshape(1, 1, 3)
        # self.radius = (self.scene_bbox[1] - self.center).reshape(1, 1, 3)
        self.radius = tf.reshape(self.scene_bbox[1] - self.center, (1, 1, 3))

        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self) -> None:
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / tf.norm(self.directions, axis=-1, keepdims=True)
        self.intrinsics = tf.constant([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]], dtype=tf.float32)

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample=1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = tf.convert_to_tensor(pose, dtype=tf.float32)
            self.poses.append(c2w)

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths.append(image_path)
            img = Image.open(image_path)
            
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (h, w, 4)
            img = tf.reshape(img, (4, -1))
            img = tf.transpose(img)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs.append(img)

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays.append(tf.concat([rays_o, rays_d], axis=1))  # (h*w, 6)

        self.poses = tf.stack(self.poses)
        if not self.is_stack:
            self.all_rays = tf.concat(self.all_rays, axis=0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = tf.concat(self.all_rgbs, axis=0)  # (len(self.meta['frames])*h*w, 3)

        else:
            self.all_rays = tf.stack(self.all_rays, axis=0)  # (len(self.meta['frames']),h*w, 3)
            self.all_rgbs = tf.reshape(tf.stack(self.all_rgbs, axis=0), [-1, *self.img_wh[::-1], 3])  # (len(self.meta['frames']),h,w,3)

    def define_transforms(self) -> None:
        # ToTensor equivalent in tensorflow
        self.transform = lambda x: tf.convert_to_tensor(np.array(x), dtype=tf.float32) / 255.0
        
    def define_proj_mat(self) -> None:
        """
        self.intrinsics (torch.Tensor)
        self.poses (torch.Tensor)
        """
        self.proj_mat = tf.linalg.matmul(tf.expand_dims(self.intrinsics, axis=0), tf.linalg.inv(self.poses)[:,:3])
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample

    def as_dataset(self):
        return tf.data.Dataset.from_tensor_slices((self.all_rays, self.all_rgbs))
    
if __name__ == '__main__':
    dataset = BlenderDatasetTF(datadir='./data/nerf_synthetic/lego/',)
    all_rgbs = dataset.all_rgbs
    all_rays = dataset.all_rgbs

    print(all_rgbs[0])
    print(all_rays[0])
    # dataset = blender_dataset.as_dataset()    
    # # 첫번째 배치를 가져옴
    # for batch in dataset.take(1):
    #     rays, rgbs = batch

    #     # 데이터를 출력
    #     print("rays: ", rays)
    #     print("rgbs: ", rgbs)