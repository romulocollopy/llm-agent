from pathlib import Path


def stringify_content(content, acc=None):
    acc = [] if acc is None else acc

    if isinstance(content, list):
        for c in content:
            stringify_content(c, acc)

    if isinstance(content, dict):
        for c in content.values():
            stringify_content(c, acc)

    if isinstance(content, str):
        acc.append(content)

    return "\n".join(acc)


def read_ignore_file(path: Path, ignore_files: set):
    if path.exists() and path.is_file():
        with open(path, "r") as file:
            lines = file.readlines()
            for line in lines:
                ignore_files.add(line.strip())


ignore_extensions = {".lock", ".db", "pyc"}
ignore_folders = {
    ".git",
    "node_modules",
    ".env",
    "chromium",
    "sqlite.db",
    "uv.lock",
    "venv",
    ".venv",
    "target",
    "dist",
    "output",
    "lib",
}


def should_ignore_file(file_path: Path, ignore_files: set) -> bool:
    return (
        file_path.name in ignore_files
        or file_path.suffix in ignore_extensions
        or any(folder in file_path.parts for folder in ignore_folders)
    )
