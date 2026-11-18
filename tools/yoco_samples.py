"""Pack fixed-length records following the pinned YOCO LMLoader._read_from_files."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow.parquet as pq

from runtime.model import tokenizer
from runtime.train import atomic_json, sha256


def document_segments(text, tok):
    # Upstream filters lines, preserving the appropriate newline suffix.
    paragraphs = text.split('\n\n')
    segments = []
    for i, paragraph in enumerate(paragraphs):
        lines = paragraph.split('\n')
        for j, line in enumerate(lines):
            if not line or len(line) > 200000:
                continue
            suffix = '\n' if j + 1 < len(lines) else ('\n\n' if i + 1 < len(paragraphs) else '')
            segments.append(tok.encode(line + suffix))
    if segments:
        segments[-1].append(tok.eos_id)
    return segments


class SamplePacker:
    def __init__(self, tok, length):
        self.tok = tok
        self.width = length + 1
        self.pending = [tok.bos_id]

    def add(self, segments):
        for ids in segments:
            room = self.width - len(self.pending)
            if len(ids) > room:
                # Upstream emits on strict overflow, drops the remainder of
                # this line, and drops the pending tail at each source EOF.
                self.pending.extend(ids[:room])
                yield self.pending
                self.pending = [self.tok.bos_id]
            else:
                self.pending.extend(ids)


def tokenize_packed_shard(task):
    raw, name, expected, revision, directory, length = task
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / (name + '.json')
    identity = dict(source_sha256=expected, revision=revision, sequence_length=length,
                    packer_sha256=sha256(__file__))
    splits = ('train', 'validation')
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if all(saved.get(k) == v for k, v in identity.items()) and all(
            Path(saved[s]['path']).is_file() and sha256(saved[s]['path']) == saved[s]['sha256']
            for s in splits
        ):
            return saved

    tok = tokenizer()
    packers = {s: SamplePacker(tok, length) for s in splits}
    documents = dict.fromkeys(splits, 0)
    report = dict(identity, source_file=str(raw))
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        with ExitStack() as stack:
            streams = {s: stack.enter_context((Path(staging) / s).open('wb')) for s in splits}
            for batch in pq.ParquetFile(raw).iter_batches(batch_size=256, columns=['id', 'text']):
                for row in batch.to_pylist():
                    segments = document_segments(row['text'] or '', tok)
                    if not segments:
                        continue
                    digest = hashlib.sha256(row['id'].encode()).digest()
                    split = 'validation' if int.from_bytes(digest[:8], 'big') % 1000 == 0 else 'train'
                    documents[split] += 1
                    for sample in packers[split].add(segments):
                        streams[split].write(np.asarray(sample, dtype='<u4').tobytes())
        for split in splits:
            dest = directory / f'{name}.{split}.bin'
            os.replace(Path(staging) / split, dest)
            report[split] = dict(path=str(dest), tokens=dest.stat().st_size // 4,
                                 documents=documents[split], sha256=sha256(dest))
    atomic_json(metadata, report)
    return report

def prepare_packed_manifest(tasks, output, length, workers, data):
    output.mkdir(parents=True, exist_ok=True)
    directory = output / sha256(__file__)
    jobs = [(*task, directory, length) for task in tasks]
    with (output / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(tokenize_packed_shard, jobs))
        splits = ('train', 'validation')
        totals = {s: sum(row[s]['tokens'] for row in records) for s in splits}
        if not all(totals.values()):
            raise ValueError('Both splits must contain complete samples')
        manifest = dict(ready=True, dataset=data['dataset'], revision=data['revision'], subset=data['subset'],
                        packing='yoco_samples', sequence_length=length, dtype='little_endian_uint32',
                        tokenizer='YOCO cl100k_base wrapper', original_yoco_data=False,
                        split_rule='sha256(id) modulo 1000 on the first 8 big-endian bytes; zero is validation',
                        source_unit='one parquet shard per split', packer_sha256=sha256(__file__),
                        shards=records, totals=totals)
        atomic_json(output / 'manifest.json', manifest)
    print(json.dumps(dict(event='data_ready', manifest=str(output / 'manifest.json'), totals=totals)))
