from scripts.compute_check import inspect_compute


def test_compute_check_has_expected_fields():
    info = inspect_compute()
    assert info["cpu_count"] is None or info["cpu_count"] > 0
    assert info["local_disk_free_gb"] >= 0
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["gpus"], list)
def test_colab_detected_in_fresh_subprocess_context(monkeypatch):
    from scripts import compute_check
    # Colab installs its package but a new CLI process has not imported it yet.
    monkeypatch.setattr(compute_check.Path, "exists", lambda self: str(self) == "/content")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_BACKEND_VERSION", raising=False)
    monkeypatch.setattr(compute_check.importlib.util, "find_spec", lambda name: object())
    assert compute_check.detect_runtime() == "colab"


def test_content_folder_alone_does_not_imply_colab(monkeypatch):
    from scripts import compute_check
    monkeypatch.setattr(compute_check.Path, "exists", lambda self: str(self) == "/content")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_BACKEND_VERSION", raising=False)
    monkeypatch.setattr(compute_check.importlib.util, "find_spec", lambda name: None)
    assert compute_check.detect_runtime() == "local_or_server"
