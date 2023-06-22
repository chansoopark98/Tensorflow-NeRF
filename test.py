import tensorflow as tf

def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
) -> tf.Tensor:
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

H, W = 800, 800
focal = [1111.111, 1111.111]

# grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
result = get_ray_directions_tf(H=H, W=W, focal=focal, center=None)
print(result)
print(result.shape)
