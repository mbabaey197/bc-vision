from app.ai.model_manager import sha256_file, verify_file


def test_hash_verification(tmp_path):
    path = tmp_path / "model.bin"
    payload = b"bc-vision-model"
    path.write_bytes(payload)
    digest = sha256_file(path)
    assert verify_file(path, digest, len(payload))
    assert not verify_file(path, "0" * 64)
    assert not verify_file(path, digest, 1)
