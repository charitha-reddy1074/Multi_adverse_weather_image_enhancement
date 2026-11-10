from pathlib import Path
import random
import csv
from collections import Counter
from datasets.pairing import image_map
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PairedWeatherDataset(Dataset):
    def __init__(self, root, crop_size=256, augment=True, mixup_prob=0.5,
                 manifest=None,balance_weather=False):
        root = Path(root)
        self.manifest = Path(manifest) if manifest else (root if root.is_file() else None)
        self.records=[]
        if self.manifest:
            base=self.manifest.parent
            with self.manifest.open(newline="",encoding="utf8") as f:
                for row in csv.DictReader(f):
                    ip=Path(row["input_path"]); tp=Path(row["target_path"])
                    self.records.append((ip if ip.is_absolute() else base/ip,
                                         tp if tp.is_absolute() else base/tp,
                                         row.get("weather_type","unknown"),
                                         row.get("source_dataset","unknown"),
                                         row.get("split","unknown"),
                                         row.get("scene_id",ip.stem)))
        else:
            degraded_root = root / "degraded"
            if not degraded_root.is_dir(): degraded_root = root / "rainy"
            degraded, clean = image_map(degraded_root), image_map(root / "clean")
            if set(degraded) != set(clean):
                raise ValueError(f"Unpaired images under {root}: inputs={len(degraded)}, targets={len(clean)}")
            self.records=[(degraded[k],clean[k],"unknown","unknown","unknown",k)
                          for k in sorted(set(degraded)&set(clean))]
        self.keys=[r[2]+"/"+r[3]+"/"+r[5] for r in self.records]
        if not self.keys:
            raise RuntimeError(f"No paired images found using {self.manifest or root}; provide input_path and target_path pairs.")
        self.crop_size, self.augment = crop_size, augment
        self.mixup_prob = float(mixup_prob)
        labels=[r[2] for r in self.records]
        counts=Counter(labels)
        self.sample_weights=[1.0/counts[label] for label in labels] if balance_weather else None
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        input_path,target_path,*meta=self.records[idx]
        k=self.keys[idx]
        a = Image.open(input_path).convert("RGB")
        b = Image.open(target_path).convert("RGB")
        if a.size != b.size:
            raise ValueError(f"Pair size mismatch for {k}: degraded={a.size}, clean={b.size}")
        w, h = a.size
        if self.crop_size:
            # Pad small images before taking a crop so batches always collate.
            if w < self.crop_size or h < self.crop_size:
                pad_w, pad_h = max(0, self.crop_size-w), max(0, self.crop_size-h)
                a, b = TF.pad(a, (0, 0, pad_w, pad_h)), TF.pad(b, (0, 0, pad_w, pad_h))
                w, h = a.size
            left = random.randint(0, w-self.crop_size)
            top = random.randint(0, h-self.crop_size)
            box = (left, top, left+self.crop_size, top+self.crop_size)
            a, b = a.crop(box), b.crop(box)
        if self.augment:
            if random.random() < .5:
                a, b = TF.hflip(a), TF.hflip(b)
            if random.random() < .5:
                a, b = TF.vflip(a), TF.vflip(b)
            turns = random.randrange(4)
            if turns:
                a, b = TF.rotate(a, 90*turns), TF.rotate(b, 90*turns)
        a, b = TF.to_tensor(a), TF.to_tensor(b)
        if self.augment and random.random() < self.mixup_prob:
            j = random.randrange(len(self.keys))
            aa = Image.open(self.records[j][0]).convert("RGB")
            bb = Image.open(self.records[j][1]).convert("RGB")
            if aa.size != bb.size:
                raise ValueError(f"Pair size mismatch for {self.keys[j]}")
            if self.crop_size:
                aw, ah = aa.size
                if aw < self.crop_size or ah < self.crop_size:
                    aa, bb = TF.pad(aa, (0, 0, max(0,self.crop_size-aw), max(0,self.crop_size-ah))), TF.pad(bb, (0, 0, max(0,self.crop_size-aw), max(0,self.crop_size-ah)))
                    aw, ah = aa.size
                left, top = random.randint(0,aw-self.crop_size), random.randint(0,ah-self.crop_size)
                box=(left,top,left+self.crop_size,top+self.crop_size)
                aa,bb=aa.crop(box),bb.crop(box)
            aa, bb = TF.to_tensor(aa), TF.to_tensor(bb)
            if aa.shape != a.shape:
                # crop_size=None can mix arbitrary validation-sized samples only
                # when dimensions match; training callers should use fixed crops.
                metadata={"key":k,"weather_type":meta[0],"source_dataset":meta[1],
                          "split":meta[2],"scene_id":meta[3]}
                return a, b, metadata
            lam = random.random()
            a, b = lam*a + (1-lam)*aa, lam*b + (1-lam)*bb
        metadata={"key":k,"weather_type":meta[0],"source_dataset":meta[1],
                  "split":meta[2],"scene_id":meta[3]}
        return a, b, metadata
