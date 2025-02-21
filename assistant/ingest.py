#!/usr/bin/env python3

import os
from pathlib import Path
import fnmatch

from tqdm import tqdm

from ipynb_filter import convert_ipynb

ignore_directory_patterns = {
    "*/assistant",
    "*/build",
    "*/docs",
    "*/figs",
    "*/.git",
    "*/.github",
    "*/.venv",
    "*/.pytest_cache",
    "*/__pycache__",
    "*/scratch",
    "*/data",
    "*/tests",
    "*.egg-info",
}
ignore_file_patterns = {
    "*.pkl",
    "*.pyc",
    "*.jpg",
    "*.png",
    "*.gif",
    "*.yml",
    "*.toml",
    "*.env",
    "*.DS_Store",
    "*.env.leave",
    "*.gitignore",
    "__init__.py",
}


def should_skip_directory(d):
    for p in ignore_directory_patterns:
        if fnmatch.fnmatch(d, p):
            return True
    return False


def should_skip_file(fp):
    for p in ignore_file_patterns:
        if fnmatch.fnmatch(fp, p):
            return True
    return False


def walk_directory(path, callback, skip_dirs=None):

    if skip_dirs is None:
        skip_dirs = list()

    # walk directory twice so we can monitor progress
    total_dirs = 0
    for root, dirs, files in tqdm(os.walk(path, topdown=True)):
        if should_skip_directory(root) or (root in skip_dirs):
            dirs.clear()
        total_dirs += 1

    with tqdm(total=total_dirs, desc="Processing directory", unit="directory") as pbar:

        for root, dirs, files in tqdm(os.walk(path, topdown=True)):
            pbar.update(1)

            if should_skip_directory(root) or (root in skip_dirs):
                dirs.clear()

            else:

                for file_name in files:

                    if not should_skip_file(file_name):

                        callback(os.path.join(root, file_name))


def ingest(root_path, outfile, skip_dirs=None):

    output = []

    br = "-" * 80

    def append_to_output(file_path):

        with open(file_path) as f:

            rel_path = file_path.removeprefix(str(root_path))

            if file_path.endswith(".ipynb"):
                source = convert_ipynb(file_path)
            else:
                try:
                    source = f.read()
                except Exception as e:
                    print(file_path)
                    raise e

            output.append(f"\n{br}\n.{rel_path} \n\n{source}")

    walk_directory(root_path, append_to_output, skip_dirs)

    with open(outfile, "w") as f:
        f.write("\n".join(output))


def run(root_path: str):

    outdir = "knowledge"
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    repo_root = Path(root_path).expanduser().absolute()

    print("reading documentation")
    outpath = os.path.join(outdir, "docs.txt")
    ingest(
        repo_root,
        outpath,
        skip_dirs=[os.path.join(repo_root, x) for x in ["src", "examples"]],
    )
    print(f"wrote {outpath}")

    print("reading src files")
    outpath = os.path.join(outdir, "src.txt")
    ingest(os.path.join(repo_root, "src"), outpath)
    print(f"wrote {outpath}")

    print("reading examples and converting .ipynb -> .py")
    outpath = os.path.join(outdir, "examples.txt")
    ingest(os.path.join(repo_root, "examples"), outpath)
    print(f"wrote {outpath}")


if __name__ == "__main__":

    # Get the parent directory of the current file
    neuromancer_root_directory = Path(__file__).resolve().parent.parent
    run(str(neuromancer_root_directory))
