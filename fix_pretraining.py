"""Apply reviewed correctness fixes to all notebook copies of shared code."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / "work" / "audit" / "originals"
BACKUP.mkdir(parents=True, exist_ok=True)


def replace(source, old, new):
    assert source.count(old) == 1, f"Expected one occurrence: {old[:100]!r}"
    return source.replace(old, new)


for filename in ["sr_part1_clean.ipynb", "sr_part2a_recognition.ipynb", "sr_part2b1_variants.ipynb"]:
    path = ROOT / filename
    if not (BACKUP / filename).exists():
        (BACKUP / filename).write_bytes(path.read_bytes())
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "class Config:" in source:
            source = replace(source, "num_workers: int = 4        # use 0 on Windows notebooks", "num_workers: int = 0 if os.name == 'nt' else 4  # Windows notebooks need in-process loading")
        if "class Degrader:" in source:
            source = replace(source,
                '        ok, buf = cv2.imencode(".jpg", lr, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*self.jpeg))])    # 4) JPEG compression artifacts\n        return cv2.imdecode(buf, cv2.IMREAD_COLOR)',
                '        bgr = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)\n        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*self.jpeg))])\n        if not ok: raise RuntimeError("JPEG encoding failed")\n        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)\n        if decoded is None: raise RuntimeError("JPEG decoding failed")\n        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)  # preserve RGB and use correct JPEG colour transform')
        if "def to_tensor(a):" in source:
            source = replace(source, "torch.from_numpy(np.ascontiguousarray(a))", "torch.from_numpy(np.array(a, copy=True, order='C'))")
        if "class FaceTrainSet(Dataset):" in source:
            source = replace(source,
                '    def __init__(self, path, ids, degrader, hr_size):\n        self.path, self.ids, self.deg, self.s, self.hr = path, ids, degrader, hr_size, None\n    def __len__(self): return len(self.ids)',
                '    def __init__(self, path, ids, degrader, hr_size, seed=42):\n        self.path, self.ids, self.deg, self.s, self.hr = path, ids, degrader, hr_size, None\n        self.seed = int(seed)\n        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()\n    def set_epoch(self, epoch): self._epoch.fill_(int(epoch))\n    def __len__(self): return len(self.ids)')
            source = replace(source,
                '        img, rng = np.asarray(self.hr[self.ids[i]]), np.random.default_rng()',
                '        img = np.asarray(self.hr[self.ids[i]])\n        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(self._epoch.item()), int(self.ids[i])]))')
            source = replace(source,
                'FaceTrainSet(cache.path, train_ids, deg, cfg.hr_size)',
                'FaceTrainSet(cache.path, train_ids, deg, cfg.hr_size, seed=cfg.seed)')
            source = replace(source,
                'persistent_workers=cfg.num_workers > 0)',
                'persistent_workers=cfg.num_workers > 0, generator=torch.Generator().manual_seed(cfg.seed))')
        if "class Trainer:" in source:
            source = replace(source,
                '                    "epoch": self.epoch, "best": self.best_psnr, "hist": self.history}, path)',
                '                    "epoch": self.epoch, "best": self.best_psnr, "hist": self.history,\n                    "config": dict(vars(self.cfg)), "protocol": "rgb_jpeg_seeded_v2",\n                    "torch_rng": torch.get_rng_state(),\n                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []}, path.with_suffix(".tmp"))\n        path.with_suffix(".tmp").replace(path)  # atomic replacement after a complete write')
            source = replace(source,
                '        ck = torch.load(self.last, map_location=device)\n        self.model.load_state_dict',
                '        ck = torch.load(self.last, map_location=device)\n        if ck.get("protocol") != "rgb_jpeg_seeded_v2" or ck.get("config") != dict(vars(self.cfg)):\n            raise ValueError("Checkpoint configuration/protocol mismatch; use a new run directory")\n        torch.set_rng_state(ck["torch_rng"].cpu())\n        if device == "cuda" and ck.get("cuda_rng"): torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])\n        self.model.load_state_dict')
            source = replace(source,
                '        for ep in range(self.epoch, self.cfg.epochs):\n            self.model.train();',
                '        for ep in range(self.epoch, self.cfg.epochs):\n            if hasattr(self.loader.dataset, "set_epoch"): self.loader.dataset.set_epoch(ep)\n            if self.loader.generator is not None: self.loader.generator.manual_seed(self.cfg.seed + ep)\n            self.model.train();')
            source = replace(source,
                '                self.opt.zero_grad(set_to_none=True); loss.backward()',
                '                if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")\n                self.opt.zero_grad(set_to_none=True); loss.backward()')
            source = replace(source,
                'nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)',
                'nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)')
        if "class EmbeddingBank:" in source:
            source = replace(source,
                '        if f.exists():\n            z = np.load(f)\n            if len(z["files"]) == len(files) and z["files"][0] == files[0] and z["files"][-1] == files[-1]:\n                print("loaded cached embeddings:", tag); return {c: z[c] for c in self.CONDS}',
                '        import hashlib\n        digest = hashlib.sha256(b"RGB-112-LR28-ArcFace-v2")\n        digest.update(str(lr_size).encode())\n        digest.update(Path(__file__).read_bytes())\n        for name in files:\n            p = Path(paths[name]); stat = p.stat()\n            digest.update(f"{name}|{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())\n        for model in [self.arc, *self.models.values()]:\n            for key, value in model.state_dict().items():\n                digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())\n        fingerprint = digest.hexdigest()\n        if f.exists():\n            with np.load(f) as z:\n                if "fingerprint" in z and str(z["fingerprint"]) == fingerprint and np.array_equal(z["files"], np.asarray(files)):\n                    print("loaded cached embeddings:", tag); return {c: z[c] for c in self.CONDS}')
            source = replace(source,
                '        np.savez(f, files=np.array(files), **emb); return emb',
                '        np.savez(f, files=np.array(files), fingerprint=np.array(fingerprint), **emb); return emb')
        if "class VerificationEval:" in source:
            source = replace(source,
                '        i = int(np.argmax(neg_below + pos_from))                      # accept pairs with sorted index >= i',
                '        boundaries = np.r_[0, np.flatnonzero(np.diff(s) > 0) + 1, len(s)]\n        i = int(boundaries[np.argmax((neg_below + pos_from)[boundaries])])  # tied scores cannot be separated')
        cell["source"] = source.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Updated", filename)
