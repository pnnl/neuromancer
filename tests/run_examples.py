import os
base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')

failed_examples = []


def run(path, failed_examples=failed_examples):
    print(f'Running example scripts in folder {path}')
    dirs = [os.path.join(path, k) for k in os.listdir(path) if os.path.isdir(os.path.join(path, k)) and k!= 'mlruns' and k != 'figs']
    files = [k for k in os.listdir(path) if k.endswith('.py') and k != 'runall.py']
    for f in files:
        print(f'Running {f}')
        status = os.system(f'python {os.path.join(path, f)} -epochs 1 >> results.txt')
        print(f'{f} exited with status={status}')
        if status !=0:
            failed_examples += [f]
            # for f in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            #     if (not f.endswith('.py')) and (not f.endswith('.pkl') and not f.startswith('psl') and not f.startswith('slim')):
            #         os.system(f'rm -rf {f}')
    for d in dirs:
        failed_examples += run(d, failed_examples)
        # for f in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        #     if not f.endswith('.py') and not f.endswith('.pkl') and not f.startswith('psl') and not f.startswith('slim'):
        #         os.system(f'rm -rf {f}')
    return failed_examples


def test_examples():
    failed_examples = []
    failed_examples = set(run(base, failed_examples=failed_examples))
    assert len(failed_examples) == 0, f'Failed examples: {failed_examples}'


if __name__ == "__main__":
    test_examples()
