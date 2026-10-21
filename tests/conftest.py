import pytest
from sepsisguard.synthetic import make_demo


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    target = tmp_path_factory.mktemp("physionet_demo") / "synthetic"
    make_demo(target, n_per_source=80)
    return target
