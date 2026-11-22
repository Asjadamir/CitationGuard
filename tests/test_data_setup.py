import hashlib
import io
import tarfile

import pytest

from scripts import setup_data


def test_real_checksum_rejects_corrupt_archive_without_writes(tmp_path):
    with pytest.raises(ValueError, match="checksum mismatch"):
        setup_data.install_archive(b"not the official SciFact archive", tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_installs_only_named_files_not_archive_paths(tmp_path, monkeypatch):
    # Trust boundary double: a tiny archive with explicit content and a traversal.
    # Checks actual installed bytes and side effects, not strings in source code.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in (*setup_data.FILES, "../../escaped"):
            content = b'{"source": "literal fixture"}\n'
            info = tarfile.TarInfo(f"data/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    content = buffer.getvalue()
    monkeypatch.setattr(setup_data, "ARCHIVE_SHA256", hashlib.sha256(content).hexdigest())
    destination = tmp_path / "installed"
    setup_data.install_archive(content, destination)
    assert sorted(path.name for path in destination.iterdir()) == sorted(setup_data.FILES)
    assert (destination / "corpus.jsonl").read_bytes() == b'{"source": "literal fixture"}\n'
    assert not (tmp_path / "escaped").exists()
