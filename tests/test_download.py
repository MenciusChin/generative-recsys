import hashlib
import importlib.util
import re

import pytest

from generative_recsys.data import download
from generative_recsys.data.download import (
    AMAZON_MIRRORS,
    DATASETS,
    ChecksumError,
    DownloadError,
    RemoteFile,
    download_dataset,
    fetch,
)
from generative_recsys.utils.paths import REPO_ROOT

CONTENT = b"user,item,timestamp\n1,10,100\n1,11,200\n" * 50


@pytest.fixture
def source(tmp_path):
    """A 'remote' file served over file://, which urllib reads through the same code path."""
    path = tmp_path / "remote" / "toy.csv"
    path.parent.mkdir()
    path.write_bytes(CONTENT)
    return path


def remote(source, **overrides) -> RemoteFile:
    fields = {
        "name": "toy.csv",
        "urls": (source.as_uri(),),
        "sha256": hashlib.sha256(CONTENT).hexdigest(),
        "size": len(CONTENT),
    } | overrides
    return RemoteFile(**fields)


def test_fresh_download_is_verified_and_complete(tmp_path, source):
    dest = tmp_path / "raw"
    path = fetch(remote(source), dest)
    assert path == dest / "toy.csv"
    assert path.read_bytes() == CONTENT
    assert list(dest.iterdir()) == [path]  # no .part left behind


def test_rerun_skips_files_already_valid(tmp_path, source, capsys):
    dest = tmp_path / "raw"
    fetch(remote(source), dest)
    source.unlink()  # a second download would now fail, so passing proves it was skipped
    assert fetch(remote(source), dest).read_bytes() == CONTENT
    assert "already downloaded" in capsys.readouterr().out


def test_bad_checksum_fails_loudly_and_leaves_nothing(tmp_path, source):
    dest = tmp_path / "raw"
    with pytest.raises(DownloadError, match=r"checksum mismatch: expected sha256 0{64}"):
        fetch(remote(source, sha256="0" * 64), dest)
    assert list(dest.iterdir()) == []


@pytest.mark.parametrize(
    ("size", "message"),
    [(len(CONTENT) + 1, "size mismatch"), (len(CONTENT) - 1, "larger than the expected")],
)
def test_wrong_size_fails(tmp_path, source, size, message):
    with pytest.raises(DownloadError, match=message):
        fetch(remote(source, size=size), tmp_path / "raw")


def test_existing_corrupt_file_raises_until_forced(tmp_path, source):
    dest = tmp_path / "raw"
    dest.mkdir()
    (dest / "toy.csv").write_bytes(CONTENT[:-1])  # e.g. truncated by hand

    with pytest.raises(ChecksumError, match="--force"):
        fetch(remote(source), dest)
    assert fetch(remote(source), dest, force=True).read_bytes() == CONTENT


def test_falls_back_to_the_next_url(tmp_path, source):
    missing = (tmp_path / "remote" / "gone.csv").as_uri()
    path = fetch(remote(source, urls=(missing, source.as_uri())), tmp_path / "raw")
    assert path.read_bytes() == CONTENT


def test_error_lists_every_failed_url(tmp_path):
    urls = tuple((tmp_path / name).as_uri() for name in ("a.csv", "b.csv"))
    with pytest.raises(DownloadError) as excinfo:
        fetch(remote(tmp_path, urls=urls), tmp_path / "raw")
    assert all(url in str(excinfo.value) for url in urls)


def test_interrupted_download_leaves_no_partial_file(tmp_path, source, monkeypatch):
    def interrupted(url, part, file):
        part.write_bytes(CONTENT[:10])
        raise KeyboardInterrupt

    monkeypatch.setattr(download, "_download", interrupted)
    dest = tmp_path / "raw"
    with pytest.raises(KeyboardInterrupt):
        fetch(remote(source), dest)
    assert list(dest.iterdir()) == []


def test_download_dataset_puts_files_in_a_dataset_folder(tmp_path, source):
    other = source.with_name("other.csv")
    other.write_bytes(CONTENT)
    registry = {"toy": (remote(source), remote(other, name="other.csv"))}

    paths = download_dataset("toy", tmp_path / "raw", registry=registry)

    assert paths == [tmp_path / "raw" / "toy" / "toy.csv", tmp_path / "raw" / "toy" / "other.csv"]
    with pytest.raises(ValueError, match="Unknown dataset 'nope'"):
        download_dataset("nope", tmp_path / "raw", registry=registry)


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_registry_is_well_formed(dataset):
    files = DATASETS[dataset]
    assert len({f.name for f in files}) == len(files)
    for f in files:
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256)
        assert f.size > 0
        assert f.urls
        assert all(url.startswith("https://") and url.endswith("/" + f.name) for url in f.urls)


def test_amazon_files_try_ucsd_before_snap():
    for f in DATASETS["beauty"]:
        assert [url.removesuffix(f.name) for url in f.urls] == list(AMAZON_MIRRORS)
    assert "mcauleylab.ucsd.edu" in AMAZON_MIRRORS[0]


def load_script():
    path = REPO_ROOT / "scripts" / "download.py"
    spec = importlib.util.spec_from_file_location("download_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_reports_failures_with_exit_code_1(monkeypatch, capsys):
    script = load_script()

    def fail(*args, **kwargs):
        raise ChecksumError("checksum mismatch")

    monkeypatch.setattr(script, "download_dataset", fail)
    assert script.main(["--dataset", "beauty"]) == 1
    assert "error: checksum mismatch" in capsys.readouterr().err


def test_script_rejects_unknown_datasets():
    with pytest.raises(SystemExit) as excinfo:
        load_script().main(["--dataset", "netflix"])
    assert excinfo.value.code == 2
