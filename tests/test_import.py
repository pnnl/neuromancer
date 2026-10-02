import subprocess
import sys


def test_import_without_ipython():
    # IPython is a notebook dependency; a None entry in sys.modules makes
    # any import of it raise ImportError in the child interpreter.
    code = "import sys; sys.modules['IPython'] = None; import neuromancer"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
