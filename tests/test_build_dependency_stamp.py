from scripts.build_dependency_stamp import (
    dependency_fingerprint,
    stamp_matches,
    write_stamp,
)


def test_dependency_stamp_changes_only_after_lock_change(tmp_path):
    first = tmp_path / "requirements-lock.txt"
    second = tmp_path / "requirements-ai-lock.txt"
    stamp = tmp_path / "cache" / "dependencies.sha256"
    first.write_text("one==1\n", encoding="utf-8")
    second.write_text("two==2\n", encoding="utf-8")

    assert not stamp_matches(stamp, [first, second])
    write_stamp(stamp, [first, second])
    assert stamp_matches(stamp, [first, second])
    assert len(dependency_fingerprint([first, second])) == 64

    second.write_text("two==3\n", encoding="utf-8")
    assert not stamp_matches(stamp, [first, second])
