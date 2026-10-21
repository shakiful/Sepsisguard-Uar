from zipfile import ZipFile
from sepsisguard.download import extract_archive


def test_zip_file_structure_is_flattened_and_safe(tmp_path):
    zip_path = tmp_path / 'training_setA.zip'
    with ZipFile(zip_path, 'w') as z:
        z.writestr('training/p001001.psv', 'HR|SepsisLabel\n90|0\n')
        z.writestr('training/p001002.psv', 'HR|SepsisLabel\n100|1\n')
        z.writestr('README.txt', 'other files ignored')
    target = tmp_path / 'training_setA'
    assert extract_archive(zip_path, target) == 2
    assert sorted(p.name for p in target.glob('*.psv')) == ['p001001.psv', 'p001002.psv']
    assert extract_archive(zip_path, target) == 2  # idempotent
