"""Pinned checkpoints used in final.pdf; model weights stay outside the submission."""
from pathlib import Path
from huggingface_hub import snapshot_download

BCE_REPO = 'maidalun1020/bce-embedding-base_v1'
BCE_REVISION = 'f542e557e78bd8c5feed08573f183d87bc3d5535'
CLIP_REPO = 'openai/clip-vit-base-patch32'
CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'

def resolve_model(repo, revision, path=None, download=False):
    if path is not None:
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    patterns = ['*.json','*.txt','*.model','*.safetensors']
    if repo == BCE_REPO:
        patterns.append('pytorch_model.bin')
    else:
        patterns += ['pytorch_model.bin','merges.txt','vocab.json']
    try:
        return Path(snapshot_download(repo_id=repo, revision=revision, local_files_only=not download,
                                      allow_patterns=patterns))
    except Exception as exc:
        raise RuntimeError(f'{repo} weights unavailable. Use --download-model or supply its local model path.') from exc
