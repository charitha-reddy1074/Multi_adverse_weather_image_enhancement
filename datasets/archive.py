"""Extract data archives without executing code or overwriting existing files."""
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PureWindowsPath


def safe_extract(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def target(name):
        win = PureWindowsPath(name)
        path = (destination / name.replace("\\", "/")).resolve()
        if win.drive or win.is_absolute() or ":" in name or not path.is_relative_to(destination):
            raise ValueError(f"Unsafe archive member: {name}")
        return path

    def copy(stream, path, size):
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Refusing to overwrite existing path: {path}")
            # Existing members must match, even when size matches.
            with path.open("rb") as current:
                while chunk := stream.read(1024 * 1024):
                    if current.read(len(chunk)) != chunk:
                        raise ValueError(f"Existing member differs: {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            shutil.copyfileobj(stream, output, length=1024 * 1024)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            # Validate all paths/types before writing anything.
            for member in members:
                target(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Archive symlink: {member.filename}")
            for member in members:
                path = target(member.filename)
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    with source.open(member) as stream:
                        copy(stream, path, member.file_size)
    else:
        with tarfile.open(archive, "r:*") as source:
            members = source.getmembers()
            for member in members:
                target(member.name)
                if not (member.isdir() or member.isfile()):
                    raise ValueError(f"Unsupported archive member: {member.name}")
            for member in members:
                path = target(member.name)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    with source.extractfile(member) as stream:
                        copy(stream, path, member.size)
