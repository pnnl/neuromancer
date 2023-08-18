import neuromancer as nm
import os


def test_directory():
    nm_path1 = nm.__path__[0].removesuffix(os.path.join('src', 'neuromancer'))
    nm_path2 =    os.getcwd().removesuffix(os.path.join('tests', 'psl'))
    assert os.path.normcase(nm_path1)==os.path.normcase(nm_path2), 'installed neuromancer root directory not the same as test root directory'
