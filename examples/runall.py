import os
base = os.path.dirname(os.path.abspath(__file__))

failed_examples = []

# status = 0
def run(path, failed_examples=failed_examples):
    print(f'Running example scripts in folder {path}')
    dirs = [k for k in os.listdir(path) if os.path.isdir(k) if k!= 'mlruns' and k != 'figs']
    files = [k for k in os.listdir(path) if k.endswith('.py') and k != 'runall.py']
    for f in files:
        status = os.system(f'python {os.path.join(path, f)} -epochs 1 >> results.txt')
        print(f'{f} exited with status={status}')
        if status !=0:
            failed_examples += [f]
        os.system('rm -rf *.png; rm -rf test*; rm -rf mlruns; rm -rf *.png')
    for d in dirs:
        failed_examples += run(os.path.join(path, d), failed_examples)
    return failed_examples


if __name__ == '__main__':
    failed_examples = []
    failed_examples = run(base, failed_examples=failed_examples)
    print(set(failed_examples))
