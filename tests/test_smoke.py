import riskcheck


def test_package_imports() -> None:
    assert riskcheck.__version__
    assert isinstance(riskcheck.SEED, int)
