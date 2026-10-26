"""Download raw dataset files and verify them against pinned sizes and SHA-256 checksums.

Files land in ``data/raw/<dataset>/`` under their upstream names. A file at its final
path is always complete: downloads are written to ``<name>.part`` and renamed only
after the size and checksum match, so an interrupted download never looks finished.

Sources, licences and where each checksum came from are documented in ``data/README.md``.
"""

import hashlib
import http.client
import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from generative_recsys.utils.paths import RAW_DATA_DIR

CHUNK_SIZE = 1 << 20  # 1 MiB
TIMEOUT_S = 60  # per network operation (connect, each read), not the whole download


class DownloadError(RuntimeError):
    """No source produced a valid copy of a file."""


class ChecksumError(DownloadError):
    """A file's size or SHA-256 doesn't match the pinned value."""


@dataclass(frozen=True)
class RemoteFile:
    """One raw file: its upstream name, where to get it and what its bytes must hash to."""

    name: str
    urls: tuple[str, ...]  # tried in order until one gives a valid file
    sha256: str
    size: int  # bytes; checked before hashing, so a truncated file fails fast


GROUPLENS = "https://files.grouplens.org/datasets/movielens/"
# The Amazon 2014 files: the UCSD McAuley Lab copy first, then the original SNAP host.
AMAZON_MIRRORS = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/",
)


def _amazon(name: str, sha256: str, size: int) -> RemoteFile:
    return RemoteFile(name, tuple(base + name for base in AMAZON_MIRRORS), sha256, size)


DATASETS: dict[str, tuple[RemoteFile, ...]] = {
    "beauty": (
        _amazon(
            "reviews_Beauty_5.json.gz",
            sha256="6b4314ec0d74fba455d4ada65cf1313194665797c0f2d33a7c9a516da3968c20",
            size=44_786_479,
        ),
        _amazon(
            "meta_Beauty.json.gz",
            sha256="de77f42de70131ef35cb8e3274d4a290e511203c2cd0396b6440d541da055095",
            size=97_828_609,
        ),
    ),
    "ml-1m": (
        RemoteFile(
            "ml-1m.zip",
            urls=(GROUPLENS + "ml-1m.zip",),
            # GroupLens publishes MD5 c4d9eecfca2ab87c1945afe126590906 (ml-1m.zip.md5);
            # this SHA-256 is of a download that matched it.
            sha256="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
            size=5_917_549,
        ),
    ),
}


def download_dataset(
    dataset: str,
    data_dir: str | Path = RAW_DATA_DIR,
    *,
    force: bool = False,
    registry: Mapping[str, tuple[RemoteFile, ...]] = DATASETS,
) -> list[Path]:
    """Make sure every raw file of ``dataset`` is in ``data_dir/<dataset>/`` and valid.

    Files already present with the right checksum are skipped. A present file with
    the wrong checksum raises ``ChecksumError`` unless ``force`` is set, in which
    case every file is downloaded again.
    """
    if dataset not in registry:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(registry)}")
    dest_dir = Path(data_dir) / dataset
    return [fetch(file, dest_dir, force=force) for file in registry[dataset]]


def fetch(file: RemoteFile, dest_dir: Path, *, force: bool = False) -> Path:
    """Return the path of a verified copy of ``file`` in ``dest_dir``, downloading if needed."""
    path = dest_dir / file.name
    if path.exists() and not force:
        try:
            verify(file, path)
        except ChecksumError as e:
            raise ChecksumError(
                f"{path} exists but is not the expected file ({e}). "
                "Delete it, or re-run with --force to download it again."
            ) from e
        print(f"{file.name}: ok (already downloaded)")
        return path

    dest_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{file.name}.part")
    errors = []
    for url in file.urls:
        try:
            _download(url, part, file)
        except (OSError, http.client.HTTPException, ChecksumError) as e:
            errors.append(f"  {url}: {e}")  # URLError and timeouts are OSErrors
        else:
            os.replace(part, path)  # atomic: the final name only ever holds a verified file
            print(f"{file.name}: downloaded and verified")
            return path
        finally:
            part.unlink(missing_ok=True)  # after a failure or Ctrl-C, leave nothing behind
    raise DownloadError(f"{file.name}: no source gave a valid file\n" + "\n".join(errors))


def verify(file: RemoteFile, path: Path) -> None:
    """Raise ``ChecksumError`` unless ``path`` has ``file``'s pinned size and SHA-256."""
    _check_size(file, path.stat().st_size)  # a truncated file fails before hashing
    _check_sha256(file, _sha256(path))


def _download(url: str, part: Path, file: RemoteFile) -> None:
    """Stream ``url`` into ``part``, hashing as it goes; raise if the result isn't ``file``."""
    digest = hashlib.sha256()
    size = 0
    # Some servers reject urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "generative-recsys"})
    with (
        urllib.request.urlopen(request, timeout=TIMEOUT_S) as response,
        part.open("wb") as out,
        tqdm(total=file.size, unit="B", unit_scale=True, desc=file.name, disable=None) as bar,
    ):
        while chunk := response.read(CHUNK_SIZE):
            size += len(chunk)
            if size > file.size:  # don't fill the disk with the wrong file
                raise ChecksumError(f"larger than the expected {file.size:,} bytes")
            digest.update(chunk)
            out.write(chunk)
            bar.update(len(chunk))
    _check_size(file, size)
    _check_sha256(file, digest.hexdigest())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _check_size(file: RemoteFile, size: int) -> None:
    if size != file.size:
        raise ChecksumError(f"size mismatch: expected {file.size:,} bytes, got {size:,}")


def _check_sha256(file: RemoteFile, sha256: str) -> None:
    if sha256 != file.sha256:
        raise ChecksumError(f"checksum mismatch: expected sha256 {file.sha256}, got {sha256}")
