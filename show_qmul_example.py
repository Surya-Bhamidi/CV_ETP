"""Export actual surveillance crops, native/protocol SR and labelled verification pairs."""
import json
import shutil

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from investigate_identity import load
from refine_model import FrozenFace, sha, write_json
from run_project import ROOT, notebook_definitions, runtime

OUT = ROOT / 'work/attention_investigation'


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', size)


def tensor(image, device):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].float().to(device) / 255


def image(value):
    value = value[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    return Image.fromarray((value * 255).round().astype('uint8'))


def panel(images, path, title, subtitle):
    tile, margin = 224, 20
    canvas = Image.new('RGB', (margin + len(images) * (tile + margin), 414), '#f3f3ee')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), title, font=font(23), fill='#18291d')
    draw.text((20, 49), subtitle, font=font(15), fill='#455249')
    for index, (label, picture, caption) in enumerate(images):
        x = margin + index * (tile + margin)
        draw.text((x, 87), label, font=font(15), fill='#18291d')
        fitted = ImageOps.contain(picture, (tile, tile), Image.Resampling.NEAREST)
        canvas.paste(fitted, (x + (tile - fitted.width) // 2, 119 + (tile - fitted.height) // 2))
        draw.text((x, 355), caption, font=font(14), fill='#455249')
    canvas.save(path)


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n, _ = runtime(); torch.backends.cudnn.benchmark = False
    model_path = ROOT / 'work/identity_investigation/mixed_quality_v1/best.pt'
    model = load(n, model_path)
    r = notebook_definitions('sr_part2a_recognition.ipynb', [1, 6, 7, 8, 9, 11], 'qmul_explainer')
    qmul = r.QMULData(str(ROOT / 'work/recognition'))
    # Fixed example from the previous uniformly spaced example list, not quality-selected.
    filename = '10002_cam3_53.jpg'
    source = qmul.paths[filename]
    native = Image.open(source).convert('RGB')
    shutil.copyfile(source, OUT / 'qmul_actual_native.jpg')
    protocol = native.resize((28, 28), Image.Resampling.BICUBIC)
    protocol_sr = image(model(tensor(protocol, n.device)).clamp(0, 1))
    native_sr = image(model(tensor(native, n.device)).clamp(0, 1))
    native_sr.save(OUT / 'qmul_native_reconstructed.png')
    protocol_sr.save(OUT / 'qmul_protocol_reconstructed.png')
    panel([('Actual recorded face', native, f'{native.width} x {native.height} native pixels'),
           ('Native input, bicubic', native.resize(native_sr.size, Image.Resampling.BICUBIC), 'Enlargement only'),
           ('Native input, our SR', native_sr, f'{native_sr.width} x {native_sr.height}; x4'),
           ('Benchmark input', protocol, 'Resized to 28 x 28'),
           ('Benchmark input, our SR', protocol_sr, '112 x 112; x4')],
          OUT / 'qmul_one_example.png', 'Real QMUL-SurvFace crop: recorded pixels and reconstruction',
          'No clean high-resolution target exists here. Display magnification adds no observations. File: ' + filename)
    # Show dataset-provided pair labels; do not invent a threshold from two pairs.
    mobile = FrozenFace(ROOT / 'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    pairs = []
    for pair_index, label in [(0, 'Same person: dataset label'), (len(qmul.pair_labels) // 2, 'Different people: dataset label')]:
        names = qmul.pairs[pair_index]
        photos = [Image.open(qmul.paths[name]).convert('RGB') for name in names]
        low = torch.cat([tensor(p.resize((28, 28), Image.Resampling.BICUBIC), n.device) for p in photos])
        sr = model(low).clamp(0, 1)
        embedding = mobile(sr)
        score = (embedding[0] * embedding[1]).sum().item()
        panel([('Recorded crop A', photos[0], f'{photos[0].width} x {photos[0].height}'),
               ('Recorded crop B', photos[1], f'{photos[1].width} x {photos[1].height}'),
               ('Reconstructed A', image(sr[:1]), 'From protocol 28-pixel input'),
               ('Reconstructed B', image(sr[1:]), 'From protocol 28-pixel input')],
              OUT / ('qmul_same_person_pair.png' if pair_index == 0 else 'qmul_different_people_pair.png'),
              label, f'MobileFace cosine after SR = {score:.4f}. This is not an accuracy percentage or a calibrated decision.')
        pairs.append({'pair_index': pair_index, 'files': list(names), 'label': int(qmul.pair_labels[pair_index]),
                      'mobileface_cosine_after_sr': score, 'score_is_probability': False})
    write_json(OUT / 'qmul_example.json', {'file': filename, 'source_relative_path': source.relative_to(ROOT).as_posix(),
        'native_size': list(native.size), 'raw_copy_sha256': sha(OUT / 'qmul_actual_native.jpg'), 'source_sha256': sha(source),
        'native_sr_size': list(native_sr.size), 'protocol_sr_size': list(protocol_sr.size),
        'checkpoint_sha256': sha(model_path), 'clean_hr_reference_available': False,
        'full_camera_frame_available': False, 'example_selection': 'First fixed uniformly spaced verification example from earlier run',
        'verification_pair_examples': pairs})
    # Visualize actual trained key masks on the protocol input.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    masks, scores, offsets = [], [], []
    hooks = []
    def capture_attention(layer, args):
        value = args[0]
        masks.append(layer.key_mask(value)[0].reshape(28, 28).cpu().numpy())
        scores.append(layer.score(value).sigmoid()[0, 0].cpu().numpy())
    def capture_sampling(layer, args):
        offsets.append((layer.offset(args[0]).tanh() * layer.max_offset)[0].cpu().numpy())
    for block in model.body:
        hooks.append(block.attn.register_forward_pre_hook(capture_attention))
        hooks.append(block.spatial.register_forward_pre_hook(capture_sampling))
    model(tensor(protocol, n.device))
    for hook in hooks: hook.remove()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout='constrained')
    axes[0, 0].imshow(protocol, interpolation='nearest'); axes[0, 0].set_title('Actual QMUL -> protocol 28')
    axes[0, 1].imshow(protocol_sr, interpolation='nearest'); axes[0, 1].set_title('Mixed-quality SR (112)')
    for index, mask in enumerate(masks):
        axis = axes.flatten()[index + 2]
        axis.imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axis.set_title(f'Block {index + 1}: {int(np.rint(mask).sum())}/784 keys')
    for axis in axes.flatten(): axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle('Actual learned spatial key selection | white = selected | not landmark or identity evidence')
    fig.savefig(OUT / 'qmul_attention_masks.png', dpi=150); plt.close(fig)
    write_json(OUT / 'qmul_attention_statistics.json', {'blocks': [
        {'selected': int(np.rint(mask).sum()), 'positions': mask.size, 'scorer_min': float(score.min()),
         'scorer_max': float(score.max()), 'offset_min': float(offset.min()), 'offset_max': float(offset.max()),
         'offset_abs_mean': float(np.abs(offset).mean())} for mask, score, offset in zip(masks, scores, offsets)]})
    print('QMUL examples and actual attention maps saved to', OUT, flush=True)


if __name__ == '__main__': main()
