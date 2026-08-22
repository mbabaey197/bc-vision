import os

import pytest

from app.file_identity import (
    descriptor_file_identity,
    path_file_identity,
)


def test_descriptor_and_path_identity_match_and_distinguish_files(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with first.open("rb") as stream:
        descriptor_identity = descriptor_file_identity(
            stream.fileno(),
            details=os.fstat(stream.fileno()),
        )

    assert descriptor_identity == path_file_identity(first)
    assert descriptor_identity != path_file_identity(second)


def test_identity_survives_hardlink_and_rename(tmp_path):
    original = tmp_path / "original.bin"
    linked = tmp_path / "linked.bin"
    renamed = tmp_path / "renamed.bin"
    original.write_bytes(b"evidence")
    try:
        os.link(original, linked)
    except OSError:
        pytest.skip("hard links are unavailable")

    expected = path_file_identity(original)
    assert path_file_identity(linked) == expected

    linked.rename(renamed)
    assert path_file_identity(renamed) == expected
