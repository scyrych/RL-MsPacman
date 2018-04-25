import numpy as np


def preprocess_observation(obs):
    mspacman_color = 210 + 164 + 74
    # crop and downsize
    img = obs[1:176:2, ::2]
    # to greyscale
    img = img.sum(axis=2)
    # Improve contrast
    img[img == mspacman_color] = 0
    # normalize from -128 to 127
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80, 1)
