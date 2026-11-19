"""Image-only inference experiments; every method returns the same x4 size.

No reference image, identity label or test score is used in prediction.
"""
from torch.nn import functional as F

TRANSFORMS = [(0, False), (0, True), (2, False), (2, True),
              (1, False), (1, True), (3, False), (3, True)]


def resize(value, shape):
    return F.interpolate(value, size=shape, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)


def ensemble_predictions(model, low, counts=(2, 4, 8)):
    """Invert flips/rotations before averaging independently predicted images."""
    total = None
    for index, (rotation, flip) in enumerate(TRANSFORMS[:max(counts)], 1):
        value = low.rot90(rotation, (-2, -1))
        if flip:
            value = value.flip(-1)
        value = model(value.contiguous()).clamp(0, 1)
        if flip:
            value = value.flip(-1)
        value = value.rot90(-rotation, (-2, -1))
        total = value if total is None else total + value
        if index in counts:
            yield index, total / index


def predict(model, low, method='single'):
    if method == 'single':
        return model(low).clamp(0, 1)
    if method.startswith('ensemble'):
        count = int(method.removeprefix('ensemble'))
        if count not in {2, 4, 8}:
            raise ValueError('Choose 2, 4 or 8 ensemble passes')
        return next(ensemble_predictions(model, low, (count,)))[1]
    output = model(low).clamp(0, 1)
    if method.startswith('feedback'):
        count = int(method.removeprefix('feedback'))
        if count not in {2, 3, 5, 8}:
            raise ValueError('Choose 2, 3, 5 or 8 feedback passes')
        for _ in range(count - 1):
            output = model(resize(output, low.shape[-2:]).contiguous()).clamp(0, 1)
        return output
    if method.startswith('cascade'):
        count = int(method.removeprefix('cascade'))
        if count not in {2, 3}:
            raise ValueError('Choose 2 or 3 literal upscaling passes')
        for _ in range(count - 1):
            output = model(output.contiguous()).clamp(0, 1)
        return resize(output, (low.shape[-2] * 4, low.shape[-1] * 4))
    if method == 'backprojection3':
        for _ in range(3):
            residual = low - resize(output, low.shape[-2:])
            correction = F.interpolate(residual, size=output.shape[-2:], mode='bicubic', align_corners=False)
            output = (output + .2 * correction).clamp(0, 1)
        return output
    raise ValueError('Unknown restoration pass method')
