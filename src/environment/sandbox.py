"""Reusable sandboxes: Docker (primary) or process-based (fallback for hosts
without a container runtime).  The process backend trades network/resource
isolation for portability — acceptable for self-generated repos."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import queue
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator


@dataclasses.dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    events: tuple[dict, ...] = ()


class Sandbox:
    def __init__(self, container_id: str, root: Path) -> None:
        self.container_id = container_id
        self.root = root
        self.reports = root.parent / (root.name + '-reports')
        self.command_index = 0
        self.security_events: list[str] = []

    def run(self, argv: list[str], *, timeout: float = 30, seed: int = 0) -> CommandResult:
        if not 0 < timeout <= 30:
            raise ValueError("command timeout must be in (0, 30]")
        if type(seed) is not int:
            raise TypeError("seed must be integer")
        started = time.monotonic()
        report = None
        if argv[:3] == ['python', '-m', 'pytest']:
            self.command_index += 1
            report = self.reports / f'{self.command_index}.jsonl'
            report.unlink(missing_ok=True)
            argv = ['python', '/grading/runner.py', f'/reports/{report.name}', *argv[3:]]
        command = ["docker", "exec", "--user", "65534:65534", "--workdir", "/workspace", "--env", f"PYTHONHASHSEED={seed % 4294967296}", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", self.container_id, "timeout", "--signal=KILL", f"{timeout}s", *argv]
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout + 5)
            events = []
            if report is not None and report.exists():
                try:
                    events = [json.loads(line) for line in report.read_text().splitlines()]
                    if not all(isinstance(event, dict) for event in events):
                        events = []
                except (ValueError, OSError):
                    events = []
                for event in events:
                    if event.get('event') == 'protected_write':
                        self.security_events.append('protected_asset_write:' + str(event.get('path')))
            return CommandResult(result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace"), result.returncode, time.monotonic() - started, tuple(events))
        except subprocess.TimeoutExpired as exc:
            # Restart terminates descendants if the in-container deadline failed.
            subprocess.run(["docker", "restart", self.container_id], capture_output=True, check=True)
            return CommandResult((exc.stdout or b"").decode(errors="replace"), "Command exceeded deadline", 124, time.monotonic() - started)

    def reset(self, source: Path) -> None:
        self.security_events.clear()
        self.root.chmod(0o755)
        for path in self.root.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
        for child in self.root.iterdir():
            # These are bind-mount sources: preserve their inodes across leases.
            if child.name in {'tests', 'hidden_tests'}:
                for entry in child.iterdir():
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                continue
            if child.name == 'conftest.py':
                child.write_text('')
                continue
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
        for path in source.rglob("*"):
            rel = path.relative_to(source)
            if any(part in {".git", "__pycache__", ".pytest_cache", ".hypothesis", "hidden_tests"} for part in rel.parts):
                continue
            if path.is_symlink():
                raise ValueError("repository symlinks are not allowed")
            target = self.root / rel
            if path.is_dir():
                target.mkdir(exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        for path in [self.root, *self.root.rglob("*")]:
            # Directories are not agent-writable: files cannot be swapped for
            # symlinks and new conftest files cannot be planted. Existing source
            # files remain writable through the five-tool protocol.
            protected = any(p in {'tests', 'hidden_tests'} for p in path.relative_to(self.root).parts) or path.name == 'conftest.py'
            path.chmod(0o555 if path.is_dir() else 0o444 if protected else 0o666)

    def install_hidden_tests(self, source: Path) -> None:
        """Host writes the source of the container's read-only hidden mount."""
        dest = self.root / 'hidden_tests'
        dest.chmod(0o755)
        for path in dest.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o644)
        for path in dest.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        shutil.copytree(source, dest, dirs_exist_ok=True)
        for path in [dest, *dest.rglob('*')]:
            path.chmod(0o555 if path.is_dir() else 0o444)


class SandboxPool:
    def __init__(self, image: str, *, size: int = 2, workspace: Path | None = None) -> None:
        if size < 1:
            raise ValueError("pool size must be positive")
        if "sha256:" not in image:
            raise ValueError("sandbox image must be pinned by digest/image ID")
        self.image = image
        self.size = size
        self.workspace = workspace or Path(tempfile.mkdtemp(prefix="octorl-pool-"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.available: queue.Queue[Sandbox] = queue.Queue()
        self.sandboxes: list[Sandbox] = []
        self.creation_count = 0
        try:
            for i in range(size):
                root = (self.workspace / str(i)).resolve()
                root.mkdir(parents=True, exist_ok=True)
                root.chmod(0o755)
                for name in ('tests', 'hidden_tests'):
                    (root / name).mkdir(exist_ok=True)
                (root / 'conftest.py').touch(exist_ok=True)
                assets = root.parent / (root.name + '-grading')
                assets.mkdir(exist_ok=True)
                shutil.copyfile(Path(__file__).resolve().parents[2] / 'scripts/grading_runner.py', assets / 'runner.py')
                reports = root.parent / (root.name + '-reports')
                reports.mkdir(exist_ok=True)
                reports.chmod(0o777)
                mounts = ['--volume', f'{root}:/workspace:rw', '--volume', f'{assets}:/grading:ro', '--volume', f'{reports}:/reports:rw']
                for name in ('tests', 'hidden_tests', 'conftest.py'):
                    mounts += ['--volume', f'{root / name}:/workspace/{name}:ro']
                result = subprocess.run(["docker", "run", "--detach", "--network", "none", "--cpus", "2", "--memory", "2g", "--memory-swap", "2g", "--pids-limit", "128", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777", *mounts, "--workdir", "/workspace", image, "sleep", "infinity"], capture_output=True, text=True, check=True)
                box = Sandbox(result.stdout.strip(), root)
                self.creation_count += 1
                self.sandboxes.append(box)
                warm = box.run(["python", "-c", "import pytest, hypothesis; print(pytest.__version__)"])
                if warm.exit_code:
                    raise RuntimeError(f"pytest warmup failed: {warm.stderr}")
                self.available.put(box)
        except BaseException:
            self.close()
            raise

    @contextlib.contextmanager
    def lease(self, source: Path) -> Iterator[Sandbox]:
        box = self.available.get(timeout=30)
        try:
            box.reset(source)
            yield box
        finally:
            # No agent command can leave a process running between trajectories.
            subprocess.run(["docker", "exec", box.container_id, "sh", "-c", "for p in /proc/[0-9]*; do [ \"$(stat -c %u $p 2>/dev/null)\" = 65534 ] && kill -9 ${p##*/} 2>/dev/null; done; true"], capture_output=True)
            self.available.put(box)

    def close(self) -> None:
        for box in self.sandboxes:
            subprocess.run(["docker", "rm", "--force", box.container_id], capture_output=True)
            for path in [box.root, *box.root.rglob('*')]:
                if not path.is_symlink():
                    path.chmod(0o755 if path.is_dir() else 0o644)
        self.sandboxes.clear()

    def __enter__(self) -> SandboxPool:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Docker detection
# ---------------------------------------------------------------------------

def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Process-based sandbox (fallback when Docker is unavailable)
# ---------------------------------------------------------------------------

class ProcessSandbox(Sandbox):
    """Sandbox backed by host subprocesses with process-group isolation."""

    def __init__(self, root: Path) -> None:
        cid = f"process:{socket.gethostname()}:{root}"
        super().__init__(container_id=cid, root=root)
        self._runner = root.parent / (root.name + '-grading') / 'runner.py'
        self._grading_dir = root.parent / (root.name + '-grading')
        self._active_pgids: set[int] = set()

    def run(self, argv: list[str], *, timeout: float = 30, seed: int = 0) -> CommandResult:
        if not 0 < timeout <= 30:
            raise ValueError("command timeout must be in (0, 30]")
        if type(seed) is not int:
            raise TypeError("seed must be integer")
        started = time.monotonic()
        report = None
        if argv[:3] == ['python', '-m', 'pytest']:
            self.command_index += 1
            report = self.reports / f'{self.command_index}.jsonl'
            report.unlink(missing_ok=True)
            argv = [sys.executable, str(self._runner), str(report), *argv[3:]]
        else:
            argv = [sys.executable if a == 'python' else a for a in argv]
        env = {
            'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
            'HOME': os.environ.get('HOME', '/tmp'),
            'PYTHONHASHSEED': str(seed % 4294967296),
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
            'OCTORL_WORKSPACE': str(self.root),
            'OCTORL_GRADING': str(self._grading_dir),
        }
        try:
            proc = subprocess.Popen(
                argv, cwd=str(self.root), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
            self._active_pgids.add(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=timeout + 5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
                self._active_pgids.discard(proc.pid)
                return CommandResult("", "Command exceeded deadline", 124, time.monotonic() - started)
            self._active_pgids.discard(proc.pid)
            events: list[dict] = []
            if report is not None and report.exists():
                try:
                    events = [json.loads(line) for line in report.read_text().splitlines()]
                    if not all(isinstance(event, dict) for event in events):
                        events = []
                except (ValueError, OSError):
                    events = []
                for event in events:
                    if event.get('event') == 'protected_write':
                        self.security_events.append('protected_asset_write:' + str(event.get('path')))
            return CommandResult(
                stdout.decode(errors="replace"), stderr.decode(errors="replace"),
                proc.returncode, time.monotonic() - started, tuple(events))
        except OSError as exc:
            return CommandResult("", str(exc), 1, time.monotonic() - started)

    def cleanup_processes(self) -> None:
        for pgid in list(self._active_pgids):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self._active_pgids.clear()


class ProcessSandboxPool:
    """Pool of process-based sandboxes. Same API as SandboxPool (duck-typed)."""

    def __init__(self, *, size: int = 2, workspace: Path | None = None) -> None:
        if size < 1:
            raise ValueError("pool size must be positive")
        self.size = size
        self.workspace = workspace or Path(tempfile.mkdtemp(prefix="octorl-pool-"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.available: queue.Queue[ProcessSandbox] = queue.Queue()
        self.sandboxes: list[ProcessSandbox] = []
        self.creation_count = 0
        try:
            for i in range(size):
                root = (self.workspace / str(i)).resolve()
                root.mkdir(parents=True, exist_ok=True)
                root.chmod(0o755)
                for name in ('tests', 'hidden_tests'):
                    (root / name).mkdir(exist_ok=True)
                (root / 'conftest.py').touch(exist_ok=True)
                assets = root.parent / (root.name + '-grading')
                assets.mkdir(exist_ok=True)
                shutil.copyfile(
                    Path(__file__).resolve().parents[2] / 'scripts/grading_runner.py',
                    assets / 'runner.py')
                reports = root.parent / (root.name + '-reports')
                reports.mkdir(exist_ok=True)
                reports.chmod(0o777)
                box = ProcessSandbox(root)
                self.creation_count += 1
                self.sandboxes.append(box)
                warm = box.run(["python", "-c", "import pytest, hypothesis; print(pytest.__version__)"])
                if warm.exit_code:
                    raise RuntimeError(f"pytest warmup failed: {warm.stderr}")
                self.available.put(box)
        except BaseException:
            self.close()
            raise

    @contextlib.contextmanager
    def lease(self, source: Path) -> Iterator[ProcessSandbox]:
        box = self.available.get(timeout=30)
        try:
            box.reset(source)
            yield box
        finally:
            box.cleanup_processes()
            self.available.put(box)

    def close(self) -> None:
        for box in self.sandboxes:
            box.cleanup_processes()
            for path in [box.root, *box.root.rglob('*')]:
                if not path.is_symlink():
                    path.chmod(0o755 if path.is_dir() else 0o644)
        self.sandboxes.clear()

    def __enter__(self) -> ProcessSandboxPool:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pool(
    image: str | None = None, *, size: int = 2, workspace: Path | None = None,
) -> SandboxPool | ProcessSandboxPool:
    """Return a Docker pool if Docker is available, else a process pool."""
    if image is not None and docker_available():
        return SandboxPool(image, size=size, workspace=workspace)
    return ProcessSandboxPool(size=size, workspace=workspace)
