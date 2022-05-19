import os
base = os.path.dirname(os.path.abspath(__file__))


def run(path):
    print(f'Running example scripts in folder {path}')
    dirs = [k for k in os.listdir(path) if os.path.isdir(k)]
    files = [k for k in os.listdir(path) if k.endswith('.py') and k != 'runall.py']
    for f in files:
        status = os.system(f'python {os.path.join(path, f)} -epochs 1')
        print(f'{f} exited with status={status}')
        if status !=0:
            exit()
    for d in dirs:
        run(os.path.join(path, d))

run(base)