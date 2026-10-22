import subprocess
import sys


def test_import_without_ipython():
    # IPython is a notebook dependency; a None entry in sys.modules makes
    # any import of it raise ImportError in the child interpreter.
    code = "import sys; sys.modules['IPython'] = None; import neuromancer"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_without_neural_operator_dependencies():
    # neuraloperator and torch-harmonics serve only neuromancer.modules.operators, and
    # torch-harmonics publishes wheels for Linux x86_64 alone.
    code = ("import sys; sys.modules['neuralop'] = None; sys.modules['torch_harmonics'] = None; "
            "import neuromancer")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
