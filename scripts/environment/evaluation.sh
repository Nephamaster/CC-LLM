#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_CHANNEL="${TORCH_CHANNEL:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_BASE="${TORCH_INDEX_BASE:-https://mirrors.nju.edu.cn/pytorch/whl}"
WHEELHOUSE="${WHEELHOUSE:-}"
ALLOW_CPU="${ALLOW_CPU:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Avoid broken/unreachable globally configured extra indexes.
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX PIP_CONSTRAINT

die() {
    echo "ERROR: $*" >&2
    exit 1
}

check_python() {
    local python="$1" version
    version="$("$python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" \
        || die "Cannot run Python interpreter: $python"
    [[ "$version" == "3.12" ]] || die "$python uses Python $version; Python 3.12 is required. Use a Python 3.12 interpreter and move the existing incompatible environment aside before recreating it."
}

select_torch() {
    local cuda_version

    if [[ -n "$TORCH_CHANNEL" ]]; then
        CHANNEL="$TORCH_CHANNEL"
    elif ! command -v nvidia-smi >/dev/null 2>&1 || \
         ! nvidia-smi -L 2>/dev/null | grep -q '^GPU [0-9]'; then
        [[ "$ALLOW_CPU" == "1" ]] || die "NVIDIA GPU/driver unavailable; set ALLOW_CPU=1 for CPU-only installation."
        CHANNEL=cpu
    else
        cuda_version="$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version:[[:space:]]*([0-9]+\.[0-9]+).*/\1/p' | head -n 1)" || true
        [[ -n "$cuda_version" ]] || die "Cannot detect CUDA capability from nvidia-smi; set TORCH_CHANNEL explicitly."
        CHANNEL="$("$PYTHON_BIN" -B - "$cuda_version" "$TORCH_INDEX_BASE" "$WHEELHOUSE" <<'CUDA'
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

capability, index, wheelhouse = sys.argv[1:]
limit = tuple(map(int, capability.split(".")))


class Channels(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = set()

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href", "")
        name = urlparse(href).path.rstrip("/").split("/")[-1]
        if tag == "a" and re.fullmatch(r"cu\d{3,}", name):
            self.names.add(name)


if wheelhouse:
    names = set()
    for wheel in Path(wheelhouse).glob("torch-*.whl"):
        match = re.search(r"\+(cu\d{3,})(?:[.-])", wheel.name)
        if match:
            names.add(match[1])
else:
    with urlopen(index.rstrip("/") + "/", timeout=30) as response:
        parser = Channels()
        parser.feed(response.read().decode())
    names = parser.names

available = [(int(name[2:-1]), int(name[-1]), name) for name in names]
available = [item for item in available if item[:2] <= limit]
if not available:
    raise SystemExit(
        f"No CUDA wheel channel supported by driver capability {capability} "
        f"in {wheelhouse or index}; set TORCH_CHANNEL only for a known compatible build."
    )
print(max(available)[2])
CUDA
)" || die "Cannot select a CUDA channel from the configured wheel source."
        echo "NVIDIA driver CUDA capability: $cuda_version"
    fi

    [[ "$CHANNEL" == cpu || "$CHANNEL" =~ ^cu[0-9]{3,}$ ]] \
        || die "Invalid TORCH_CHANNEL=$CHANNEL; use cu followed by the CUDA version, or cpu."
}

install_environment() {
    local target="$1" requirements="$2"
    local -a pip_args

    [[ -f "$requirements" ]] || die "Requirements file does not exist: $requirements"
    if [[ ! -x "$target/bin/python" ]]; then
        "$PYTHON_BIN" -m venv "$target"
    fi
    check_python "$target/bin/python"
    if [[ -n "$WHEELHOUSE" ]]; then
        [[ -d "$WHEELHOUSE" ]] || die "WHEELHOUSE does not exist: $WHEELHOUSE"
        pip_args=(--no-index --find-links "$WHEELHOUSE")
        if compgen -G "${WHEELHOUSE%/}/pip-*.whl" >/dev/null; then
            "$target/bin/python" -m pip install "${pip_args[@]}" --upgrade pip
        else
            echo "Offline mode: no pip wheel in WHEELHOUSE; skipping pip upgrade."
        fi
    else
        pip_args=(--index-url "$PIP_INDEX_URL")
        "$target/bin/python" -m pip install "${pip_args[@]}" --upgrade pip
    fi

    (
        local wheel_dir
        local -a selected
        wheel_dir="$(mktemp -d)"
        trap 'rm -rf -- "$wheel_dir"' EXIT
        # Inspect vLLM's actual CUDA build and Torch requirements before installing either.
        "$target/bin/python" -B - "$CHANNEL" "$wheel_dir" "$PIP_INDEX_URL" "$WHEELHOUSE" \
            "$requirements" "$TORCH_INDEX_BASE" <<'SELECT'
import ast
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
import zipfile

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

channel, folder, index, wheelhouse, requirements, torch_index = sys.argv[1:]
folder = Path(folder)
normal = ["--no-index", "--find-links", wheelhouse] if wheelhouse else ["--index-url", index]
cuda = normal if wheelhouse else ["--index-url", f"{torch_index.rstrip('/')}/{channel}/"]
pip = [sys.executable, "-m", "pip"]


def versions(package):
    output = subprocess.check_output(
        pip + ["index", "versions", *cuda, "--only-binary=:all:", package], text=True
    )
    available = next(
        (line.split(":", 1)[1] for line in output.splitlines() if line.startswith("Available versions:")), ""
    )
    return [v.strip() for v in available.split(",") if v.strip().endswith("+" + channel)]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs and "data-yanked" not in attrs:
            self.items.append((attrs["href"], attrs.get("data-requires-python")))


class RemoteWheel:
    """Read wheel metadata with HTTP ranges; spool once if the mirror ignores Range."""

    def __init__(self, url):
        self.url, self.position, self.local = url, 0, None
        with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=30) as response:
            if response.status == 206:
                self.length = int(response.headers["Content-Range"].split("/")[-1])
            else:
                self.spool(response)
                self.length = self.local.stat().st_size

    def spool(self, response):
        self.local = folder / unquote(Path(urlparse(self.url).path).name)
        with self.local.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        )
        return self.position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size <= 0:
            return b""
        if not self.local:
            end = self.position + size - 1
            with urlopen(
                Request(self.url, headers={"Range": f"bytes={self.position}-{end}"}), timeout=30
            ) as response:
                if response.status == 206:
                    data = response.read()
                else:
                    self.spool(response)
        if self.local:
            with self.local.open("rb") as handle:
                handle.seek(self.position)
                data = handle.read(size)
        self.position += len(data)
        return data


def wheel_channel(version, archive):
    local = version.local or ""
    match = re.search(r"(?:^|\.)(cu\d+|cpu)(?:\.|$)", local)
    if match:
        return match[1]
    if local:  # Do not interpret ROCm/other builds as default CUDA wheels.
        return None
    if "vllm/envs.py" not in archive.namelist():
        return None
    # Official default wheels declare the build baseline here; alternate CUDA builds use +cuXXX.
    tree = ast.parse(archive.read("vllm/envs.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "VLLM_MAIN_CUDA_VERSION" and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(value, str) and re.fullmatch(r"\d+\.\d+", value):
                    return "cu" + value.replace(".", "")
    return None


tags = set(sys_tags())
python_version = ".".join(map(str, sys.version_info[:3]))
if wheelhouse:
    links = [(str(path), None) for path in Path(wheelhouse).glob("vllm-*.whl")]
else:
    page = index.rstrip("/") + "/vllm/"
    with urlopen(page, timeout=30) as response:
        parser = Links()
        parser.feed(response.read().decode())
    links = [(urldefrag(urljoin(page, url))[0], py) for url, py in parser.items]

candidates = []
for source, py in links:
    name = unquote(Path(urlparse(source).path).name)
    if not name.endswith(".whl"):
        continue
    package, version, _, wheel_tags = parse_wheel_filename(name)
    if package == "vllm" and not version.is_prerelease and tags & wheel_tags:
        if not py or SpecifierSet(py).contains(python_version):
            candidates.append((version, source))
candidates.sort(reverse=True)
torch_versions = versions("torch")
for version, source in candidates:
    print(f"Checking vLLM {version} for {channel}", file=sys.stderr, flush=True)
    reader = Path(source) if wheelhouse else RemoteWheel(source)
    with zipfile.ZipFile(reader) as archive:
        built_channel = wheel_channel(version, archive)
        if built_channel != channel:
            print(f"  skipped: CUDA build {built_channel or 'unidentified'}", file=sys.stderr, flush=True)
            continue
        metadata_path = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
    if metadata.get("Requires-Python") and not SpecifierSet(metadata["Requires-Python"]).contains(
        python_version
    ):
        continue
    dependencies = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    dependencies = [r for r in dependencies if not r.marker or r.marker.evaluate({"extra": ""})]
    torch_rules = [r.specifier for r in dependencies if r.name == "torch"]
    if not torch_rules:
        print("  skipped: missing Torch compatibility declaration", file=sys.stderr)
        continue
    for torch_version in torch_versions:
        if not all(rule.contains(torch_version) for rule in torch_rules):
            continue
        print(f"Resolving vLLM {version} + Torch {torch_version}", file=sys.stderr, flush=True)
        paths = []
        for package, spec in [("torch", SpecifierSet("==" + torch_version))] + [
            (r.name, r.specifier) for r in dependencies if r.name in {"torchvision", "torchaudio"}
        ]:
            options = (
                [v for v in versions(package) if spec.contains(v)] if package != "torch" else [torch_version]
            )
            if not options:
                break
            subprocess.run(
                pip
                + [
                    "download",
                    *cuda,
                    "--no-deps",
                    "--only-binary=:all:",
                    "--dest",
                    str(folder),
                    f"{package}=={options[0]}",
                ],
                check=True,
            )
            paths.append(str(next(folder.glob(f"{package}-{options[0]}-*.whl"))))
        else:
            reference = source if wheelhouse else str(reader.local) if reader.local else source
            subprocess.run(
                pip
                + ["download", *normal, "--no-deps", "--only-binary=:all:", "--dest", str(folder), reference],
                check=True,
            )
            vllm_wheel = str(folder / unquote(Path(urlparse(source).path).name))
            selected = paths + [vllm_wheel]
            command = pip + ["install", *normal, "--dry-run", "-r", requirements, *selected]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(result.stdout, flush=True)
            if result.returncode == 0:
                (folder / "selected.txt").write_text("\n".join(selected) + "\n")
                sys.exit(0)
            if not re.search(r"ResolutionImpossible|conflicting dependencies", result.stdout):
                raise SystemExit("Dependency resolution failed; see pip output above.")
            print("Dependency conflict; trying another matching pair.", file=sys.stderr)
raise SystemExit(
    f"No verified vLLM/Torch wheel pair for {channel}. No application packages installed. "
    "Provide matching CUDA-labelled vLLM wheels and dependencies in WHEELHOUSE, "
    "or select a supported CUDA channel. Unknown default builds are not assumed compatible."
)
SELECT
        mapfile -t selected < "$wheel_dir/selected.txt"
        "$target/bin/python" -m pip install "${pip_args[@]}" -r "$requirements" "${selected[@]}"
        echo "Installing the selected Torch wheel last"
        "$target/bin/python" -m pip install "${pip_args[@]}" "${selected[0]}"
    )
    "$target/bin/python" -m pip check

    "$target/bin/python" -B - "$CHANNEL" <<'PY'
import sys
import torch
import transformers
import swift
import lm_eval
from vllm import LLM

channel = sys.argv[1]
print(f"Python: {sys.executable}; transformers={transformers.__version__}; torch={torch.__version__}")
if channel == "cpu":
    if torch.version.cuda is not None:
        raise SystemExit(f"Expected CPU Torch, got CUDA {torch.version.cuda}")
    print(f"Validated: torch={torch.__version__} (CPU)")
else:
    expected_cuda = f"{channel[2:-1]}.{channel[-1]}"
    if torch.version.cuda != expected_cuda:
        raise SystemExit(f"Expected CUDA {expected_cuda}, got {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; check GPU driver and PyTorch compatibility")
    result = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    print(f"Validated: torch={torch.__version__}, CUDA={torch.version.cuda}, "
          f"GPU={torch.cuda.get_device_name(0)}, test={result.item()}")
PY
}

target="$ROOT/.venv/evaluation"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Only Linux x86_64 is supported."
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python 3.12 not found: $PYTHON_BIN. Install it or set PYTHON_BIN to its path."
check_python "$PYTHON_BIN"
if [[ -e "$target" ]]; then
    [[ -x "$target/bin/python" ]] || die "Existing environment is incomplete: $target; move it aside before recreating it."
    check_python "$target/bin/python"
fi
select_torch
printf 'evaluation CUDA wheel index: %s/%s/\n' "${TORCH_INDEX_BASE%/}" "$CHANNEL"
install_environment "$target" "$ROOT/requirements/evaluation.txt"
printf '\nevaluation environment: %s\nHF endpoint: %s\n' "$target" "$HF_ENDPOINT"
