import ast
import json
import os
import re
import warnings
from pathlib import Path
import fnmatch

from tqdm import tqdm

# NeuroMANCER's docstrings contain LaTeX (e.g. \dot{x}, \mathcal{L}) in
# non-raw strings; ast.parse() below evaluates those escapes and warns.
warnings.filterwarnings("ignore", category=SyntaxWarning)

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

# autodoc detection
AUTODOC_MIN_DIRECTIVES = 1
AUTODOC_MAX_PROSE_LINES = 8

EXAMPLE_SPLIT_LINE_THRESHOLD = 200

# chunks longer than this exceed the embedding model's effective context and
# would only be partially embedded; split them into line-based parts
MAX_CHUNK_CHARS = 6000

# cap for implementation source carried as display payload on api chunks
MAX_IMPL_CHARS = 12000

AUTODOC_DIRECTIVE_RE = re.compile(
    r"^\s*\.\.\s+auto(function|class|module|method|data|attribute)::", re.I
)
SPHINX_OPTION_RE = re.compile(r"^\s*:\w+")
RST_UNDERLINE_CHARS = set('=-~^"')


def should_skip_directory(d):
    d = Path(d).as_posix()
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

    for root, dirs, files in tqdm(
        os.walk(path, topdown=True), desc="Processing directory", unit="directory"
    ):
        if should_skip_directory(root) or (root in skip_dirs):
            dirs.clear()
            continue

        for file_name in files:
            if not should_skip_file(file_name):
                callback(os.path.join(root, file_name))


def normalize_rel_path(file_path, root_path):
    return Path(file_path).relative_to(root_path).as_posix()


def make_record(rel_path, source_type, symbol_name, start_line, end_line, content):
    return {
        "id": f"{rel_path}:{symbol_name or ''}:{start_line}",
        "source_type": source_type,
        "file_path": rel_path,
        "symbol_name": symbol_name,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
    }


def read_file_text(file_path):
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def write_jsonl(records, outfile):
    with open(outfile, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_oversized(record):
    content = record["content"]
    if len(content) <= MAX_CHUNK_CHARS:
        return [record]

    lines = content.splitlines()
    parts = []
    part_lines = []
    part_start = record["start_line"]
    size = 0

    def flush(end_line):
        if not part_lines:
            return
        parts.append(
            {
                **record,
                "id": f"{record['id']}:part{len(parts)}",
                "start_line": part_start,
                "end_line": end_line,
                "content": "\n".join(part_lines),
            }
        )

    for offset, line in enumerate(lines):
        part_lines.append(line)
        size += len(line) + 1
        if size >= MAX_CHUNK_CHARS:
            flush(record["start_line"] + offset)
            part_lines = []
            part_start = record["start_line"] + offset + 1
            size = 0

    flush(record["end_line"])
    return parts


def collect_chunks(root_path, process_file, skip_dirs=None):
    records = []
    root_path = Path(root_path).resolve()

    def callback(file_path):
        for record in process_file(file_path, root_path):
            records.extend(split_oversized(record))

    walk_directory(str(root_path), callback, skip_dirs)
    return records


def is_rst_underline(line):
    s = line.strip()
    return len(s) >= 2 and all(c in RST_UNDERLINE_CHARS for c in s)


def is_autodoc_shell(text):
    directive_lines = 0
    prose_lines = 0

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if AUTODOC_DIRECTIVE_RE.match(line):
            directive_lines += 1
            i += 1
            continue

        if SPHINX_OPTION_RE.match(line):
            i += 1
            continue

        if stripped.startswith(".. "):
            i += 1
            continue

        if is_rst_underline(line):
            i += 1
            continue

        if i + 1 < len(lines) and stripped and is_rst_underline(lines[i + 1]):
            i += 2
            continue

        prose_lines += 1
        i += 1

    return (
        directive_lines >= AUTODOC_MIN_DIRECTIVES
        and prose_lines <= AUTODOC_MAX_PROSE_LINES
    )


def find_rst_headers(lines):
    headers = []
    i = 0
    while i < len(lines) - 1:
        title = lines[i].strip()
        if title and is_rst_underline(lines[i + 1]):
            headers.append((i, title))
            i += 2
        else:
            i += 1
    return headers


def find_md_headers(lines):
    headers = []
    for i, line in enumerate(lines):
        match = re.match(r"^#{1,6}\s+(.+)$", line)
        if match:
            headers.append((i, match.group(1).strip()))
    return headers


def split_by_headers(text, file_path, headers):
    lines = text.splitlines()
    if not lines:
        return []

    if not headers:
        return [
            {
                "symbol_name": Path(file_path).stem,
                "start_line": 1,
                "end_line": max(1, len(lines)),
                "content": text,
            }
        ]

    sections = []
    if headers[0][0] > 0:
        sections.append(
            {
                "symbol_name": Path(file_path).stem,
                "start_line": 1,
                "end_line": headers[0][0],
                "content": "\n".join(lines[: headers[0][0]]),
            }
        )

    for idx, (start, title) in enumerate(headers):
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        sections.append(
            {
                "symbol_name": title,
                "start_line": start + 1,
                "end_line": end,
                "content": "\n".join(lines[start:end]),
            }
        )

    return sections


def split_doc_sections(text, file_path):
    lines = text.splitlines()
    suffix = Path(file_path).suffix.lower()

    if suffix == ".rst":
        headers = find_rst_headers(lines)
    elif suffix in (".md", ".txt"):
        headers = find_md_headers(lines)
    else:
        headers = find_md_headers(lines)
        if not headers:
            headers = find_rst_headers(lines)

    return split_by_headers(text, file_path, headers)


def chunk_doc_file(file_path, root_path):
    suffix = Path(file_path).suffix.lower()
    if suffix not in (".rst", ".md", ".txt"):
        return []

    rel_path = normalize_rel_path(file_path, root_path)
    text = read_file_text(file_path)

    if is_autodoc_shell(text):
        return []

    records = []
    for section in split_doc_sections(text, file_path):
        if not section["content"].strip():
            continue
        records.append(
            make_record(
                rel_path,
                "doc",
                section["symbol_name"],
                section["start_line"],
                section["end_line"],
                section["content"],
            )
        )
    return records


def is_main_guard(node):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if not test.comparators:
        return False
    comp = test.comparators[0]
    if isinstance(comp, ast.Constant):
        return comp.value == "__main__"
    return False


def node_end_line(node, source_lines, siblings, idx):
    if getattr(node, "end_lineno", None):
        return node.end_lineno
    if idx + 1 < len(siblings) and hasattr(siblings[idx + 1], "lineno"):
        return siblings[idx + 1].lineno - 1
    return len(source_lines)


def split_large_python_example(source_lines, rel_path):
    source = "\n".join(source_lines)
    tree = ast.parse(source)
    records = []

    for idx, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol_name = node.name
        elif is_main_guard(node):
            symbol_name = "__main__"
        else:
            continue

        start = node.lineno
        end = node_end_line(node, source_lines, tree.body, idx)
        content = "\n".join(source_lines[start - 1 : end])
        records.append(
            make_record(rel_path, "example", symbol_name, start, end, content)
        )

    return records


def get_cell_source(cell):
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    return source.strip()


def is_setup_only_code(source):
    lines = [line for line in source.splitlines() if line.strip()]
    if not lines:
        return True
    return all(
        line.lstrip().startswith(("!", "%", "#")) for line in lines
    )


def first_markdown_heading(text):
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+)$", line)
        if match:
            return match.group(1).strip()
    return None


def render_ipynb_chunk(cells):
    parts = []
    for cell_type, source in cells:
        if cell_type == "markdown":
            parts.append(source)
        elif cell_type == "code":
            parts.append(f"```python\n{source}\n```")
    return "\n\n".join(parts)


def chunk_ipynb_file(file_path, rel_path):
    with open(file_path, encoding="utf-8") as f:
        nb = json.load(f)

    kept = []
    for cell_index, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        if cell_type not in ("markdown", "code"):
            continue
        source = get_cell_source(cell)
        if not source:
            continue
        if cell_type == "code" and is_setup_only_code(source):
            continue
        kept.append((cell_index, cell_type, source))

    records = []
    buffer = []
    MIN_MARKDOWN_ONLY_CHARS = 200
    carry: list[tuple] = []

    def flush_buffer():
        nonlocal buffer, carry
        if not buffer:
            return

        has_code = any(cell_type == "code" for _, cell_type, _ in buffer)
        if not has_code:
            markdown_text = "\n\n".join(
                source for _, cell_type, source in buffer if cell_type == "markdown"
            )
            if len(markdown_text.strip()) < MIN_MARKDOWN_ONLY_CHARS:
                carry.extend(buffer)
                buffer = []
                return


        cells = carry + buffer
        carry = []
        markdown_text = "\n\n".join(
            source for _, cell_type, source in cells if cell_type == "markdown"
        )
        symbol_name = (
            first_markdown_heading(markdown_text) if markdown_text else None
        )
        start_line = cells[0][0] + 1
        end_line = cells[-1][0] + 1
        content = render_ipynb_chunk(
            [(cell_type, source) for _, cell_type, source in cells]
        )
        if content.strip():
            records.append(
                make_record(
                    rel_path, "example", symbol_name, start_line, end_line, content
                )
            )
        buffer = []

    for cell_index, cell_type, source in kept:
        if cell_type == "markdown":
            if buffer:
                flush_buffer()
            buffer.append((cell_index, cell_type, source))
        else:
            buffer.append((cell_index, cell_type, source))

    flush_buffer()
    return records


def chunk_example_file(file_path, root_path):
    rel_path = normalize_rel_path(file_path, root_path)
    if Path(file_path).suffix.lower() == ".ipynb":
        return chunk_ipynb_file(file_path, rel_path)

    text = read_file_text(file_path)
    source_lines = text.splitlines()
    line_count = len(source_lines) or 1

    if line_count <= EXAMPLE_SPLIT_LINE_THRESHOLD:
        return [
            make_record(rel_path, "example", None, 1, line_count, text)
        ]

    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return split_large_python_example(source_lines, rel_path)

    sections = split_doc_sections(text, file_path)
    records = []
    for section in sections:
        if not section["content"].strip():
            continue
        records.append(
            make_record(
                rel_path,
                "example",
                section["symbol_name"],
                section["start_line"],
                section["end_line"],
                section["content"],
            )
        )
    return records


def has_doc_or_params(node):
    doc = ast.get_docstring(node)
    has_params = (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and len(node.args.args) > 1
    )
    return bool(doc and doc.strip()) or has_params


def should_index_symbol(name, node):
    if name.startswith("__") and name != "__init__":
        return False
    if name.startswith("_"):
        return has_doc_or_params(node)
    return True


def qualified_name(parent_class, name):
    if parent_class:
        return f"{parent_class}.{name}"
    return name


def extract_signature_and_doc(node, source_lines):
    start = node.lineno
    if node.decorator_list:
        start = node.decorator_list[0].lineno

    end = node.lineno
    while end <= len(source_lines):
        line = source_lines[end - 1]
        if line.rstrip().endswith(":") and not line.strip().startswith("@"):
            break
        end += 1

    parts = list(source_lines[start - 1 : end])
    doc = ast.get_docstring(node)
    if doc:
        parts.append("")
        parts.append(doc)
    return "\n".join(parts)


def iter_src_symbols(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if should_index_symbol(node.name, node):
                yield None, node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if should_index_symbol(child.name, child):
                            yield node.name, child


def chunk_src_file(file_path, root_path):
    if not file_path.endswith(".py"):
        return []

    rel_path = normalize_rel_path(file_path, root_path)
    with open(file_path, encoding="utf-8") as f:
        source = f.read()
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        print(f"SyntaxError parsing {rel_path}")
        return []

    records = []
    for parent_class, node in iter_src_symbols(tree):
        symbol_name = qualified_name(parent_class, node.name)
        start = node.lineno
        end = node.end_lineno or start
        impl_content = "\n".join(source_lines[start - 1 : end])
        doc = ast.get_docstring(node)

        if doc and doc.strip():
            # embed only the signature + docstring
            # carry the implementation along for display instead of indexing
            # a near-duplicate chunk that competes in search
            api_content = extract_signature_and_doc(node, source_lines)
            record = make_record(
                rel_path, "api", symbol_name, start, end, api_content
            )
            if len(impl_content) > MAX_IMPL_CHARS:
                impl_content = impl_content[:MAX_IMPL_CHARS] + "\n# ... truncated ..."
            record["impl"] = impl_content
            records.append(record)
        else:
            # no docstring: a bare signature won't match queries, index the code
            records.append(
                make_record(rel_path, "impl", symbol_name, start, end, impl_content)
            )

    return records


def run(root_path: str):
    """Chunk a checkout into the three JSONL corpora, each with its own strategy."""
    outdir = Path(__file__).resolve().parent.parent / "knowledge"
    outdir.mkdir(exist_ok=True)

    repo_root = Path(root_path).expanduser().resolve()
    skip_dirs = [str(repo_root / x) for x in ["src", "examples"]]

    print("reading documentation")
    docs = collect_chunks(repo_root, chunk_doc_file, skip_dirs=skip_dirs)
    docs_path = outdir / "docs.jsonl"
    write_jsonl(docs, docs_path)
    print(f"wrote {docs_path} ({len(docs)} chunks)")

    print("reading src files")
    src = collect_chunks(repo_root / "src", chunk_src_file)
    src_path = outdir / "src.jsonl"
    write_jsonl(src, src_path)
    print(f"wrote {src_path} ({len(src)} chunks)")

    print("reading examples")
    examples = collect_chunks(repo_root / "examples", chunk_example_file)
    examples_path = outdir / "examples.jsonl"
    write_jsonl(examples, examples_path)
    print(f"wrote {examples_path} ({len(examples)} chunks)")


if __name__ == "__main__":
    import sys

    default_root = Path(__file__).resolve().parent.parent.parent
    root = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("NEUROMANCER_ROOT", str(default_root))
    )
    run(str(root))
