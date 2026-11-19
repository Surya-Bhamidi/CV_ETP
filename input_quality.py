"""Explicit degradation profiles shared by the demo and paired experiments."""
import numpy as np
from PIL import Image

PROFILE_LABELS = {
    'standard28': 'Benchmark: blur, noise and JPEG',
    'mild28': 'Mild: less blur, low noise, JPEG 85-99',
    'clean28': 'Clean: bicubic downsampling only',
}


class InputDegradation:
    def __init__(self, n, profile):
        if profile not in {*PROFILE_LABELS, 'clean56'}:
            raise ValueError('Choose a valid degradation profile.')
        self.n, self.profile = n, profile

    def __call__(self, image, rng):
        if self.profile == 'standard28':
            return self.n.Degrader()(image, rng)
        if self.profile == 'mild28':
            return self.n.Degrader(4, blur_p=.2, noise=(0, 2), jpeg=(85, 100))(image, rng)
        if self.profile == 'clean56':
            size = (image.shape[1] // 2, image.shape[0] // 2)
        else:
            size = (image.shape[1] // 4, image.shape[0] // 4)
        return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC))
