import tensorflow as tf
import tensorflow_addons as tfa

def positional_encoding(positions, freqs):
    freq_bands = 2.0 ** tf.range(freqs, dtype=tf.float32)  # (F,)
    pts = tf.reshape(positions[..., None] * freq_bands, shape=tf.concat([tf.shape(positions)[:-1], [freqs * tf.shape(positions)[-1]]], axis=0))  # (..., DF)
    pts = tf.concat([tf.math.sin(pts), tf.math.cos(pts)], axis=-1)
    return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - tf.exp(-sigma * dist)

    T = tf.math.cumprod(tf.concat([tf.ones((tf.shape(alpha)[0], 1)), 1. - alpha + 1e-10], axis=-1), axis=-1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

def grid_sample(volume, grid):
    B, D, H, W, C = volume.shape
    volume = tf.reshape(volume, [B, D, H, W, C])

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # Rescale x and y to [0, W-1/H-1].
    grid = tf.cast(grid, 'float32')
    x = 0.5 * ((grid[..., 0] + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((grid[..., 1] + 1.0) * tf.cast(max_y - 1, 'float32'))

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    Ia = tf.gather_nd(volume, tf.stack([x0, y0], axis=-1))
    Ib = tf.gather_nd(volume, tf.stack([x0, y1], axis=-1))
    Ic = tf.gather_nd(volume, tf.stack([x1, y0], axis=-1))
    Id = tf.gather_nd(volume, tf.stack([x1, y1], axis=-1))

    wa = (x1-tf.cast(x, 'float32')) * (y1-tf.cast(y, 'float32'))
    wb = (x1-tf.cast(x, 'float32')) * (tf.cast(y, 'float32')-y0)
    wc = (tf.cast(x, 'float32')-x0) * (y1-tf.cast(y, 'float32'))
    wd = (tf.cast(x, 'float32')-x0) * (tf.cast(y, 'float32')-y0)

    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return output

class AlphaGridMask(tf.keras.Model):
    def __init__(self, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.aabb = aabb  # shape (2, 3)
        self.aabbSize = self.aabb[1] - self.aabb[0]  # shape (3)
        self.invgridSize = 1.0 / self.aabbSize * 2  # shape (3)
        # channels_last format: (dim1, dim2, dim3, channels)
        self.alpha_volume = tf.reshape(alpha_volume, [1, *alpha_volume.shape, 1])  # shape (1, 128, 128, 128, 1)
        self.gridSize = tf.cast(tf.shape(alpha_volume), dtype=tf.int32)  # shape (3)

    def sample_alpha(self, xyz_sampled):
        # input xyz_sampled shape = (244945, 3)
        # self.alpha_volume (tf.Tensor) : shape (1, 128, 128, 128, 1)
        xyz_sampled = self.normalize_coord(xyz_sampled)  # shape (244945, 3)
        print(tf.reduce_mean(xyz_sampled))
        alpha_vals = grid_sample(self.alpha_volume, 
                                 tf.reshape(xyz_sampled, [1, -1, 1, 1, 3]))  # shape (1, 1, 244945, 1, 1)
        alpha_vals = tf.reshape(alpha_vals, [-1])  # shape (244945)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1

if __name__ == '__main__':
    tensor = tf.ones((244945, 3))
    aabb = tf.ones((2, 3))
    alpha_volume = tf.ones((128, 128, 128))
    grid_mask = AlphaGridMask(aabb, alpha_volume)
    output = grid_mask.sample_alpha(tensor)
    print(output.shape)
    print(tf.reduce_mean(output))