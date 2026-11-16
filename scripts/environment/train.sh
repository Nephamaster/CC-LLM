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
    local -a pip_args torch_args

    [[ -f "$requirements" ]] || die "Requirements file does not exist: $requirements"
    if [[ ! -x "$target/bin/python" ]]; then
        "$PYTHON_BIN" -m venv "$target"
    fi
    check_python "$target/bin/python"
    if [[ -n "$WHEELHOUSE" ]]; then
        [[ -d "$WHEELHOUSE" ]] || die "WHEELHOUSE does not exist: $WHEELHOUSE"
        pip_args=(--no-index --find-links "$WHEELHOUSE")
        torch_args=("${pip_args[@]}")
        if compgen -G "${WHEELHOUSE%/}/pip-*.whl" >/dev/null; then
            "$target/bin/python" -m pip install "${pip_args[@]}" --upgrade pip
        else
            echo "Offline mode: no pip wheel in WHEELHOUSE; skipping pip upgrade."
        fi
    else
        pip_args=(--index-url "$PIP_INDEX_URL")
        torch_args=(--index-url "${TORCH_INDEX_BASE%/}/$CHANNEL/")
        "$target/bin/python" -m pip install "${pip_args[@]}" --upgrade pip
    fi

    (
        local wheel_dir torch_version selected=0
        local -a wheel
        wheel_dir="$(mktemp -d)"
        trap 'rm -rf -- "$wheel_dir"' EXIT
        # pip filters the candidates for this Python/platform and lists newest first.
        "$target/bin/python" -m pip index versions "${torch_args[@]}" --only-binary=:all: torch \
            > "$wheel_dir/versions.txt"
        while IFS= read -r torch_version; do
            [[ "$torch_version" == *"+$CHANNEL" ]] || continue
            echo "Resolving ${target##*/} with torch==$torch_version"
            rm -f -- "$wheel_dir"/torch-*.whl
            # Only Torch comes from the CUDA source; all its dependencies use the normal source.
            "$target/bin/python" -m pip download "${torch_args[@]}" \
                --no-deps --only-binary=:all: --dest "$wheel_dir" "torch==$torch_version"
            wheel=("$wheel_dir"/torch-*.whl)
            if "$target/bin/python" -m pip install "${pip_args[@]}" --dry-run \
                -r "$requirements" "${wheel[0]}" 2>&1 | tee "$wheel_dir/resolve.log"; then
                selected=1
                break
            fi
            # Network/build failures are not evidence that an older Torch is needed.
            grep -Eq 'ResolutionImpossible|conflicting dependencies' "$wheel_dir/resolve.log" \
                || die "Dependency resolution failed; see the pip error above."
            echo "Dependency conflict; trying the next $CHANNEL Torch candidate."
        done < <(sed -n 's/^Available versions: //p' "$wheel_dir/versions.txt" \
            | tr ',' '\n' | sed 's/^ *//;s/ *$//')
        [[ "$selected" == 1 ]] \
            || die "No compatible requirements/Torch combination for $CHANNEL; no application packages installed."

        # Use the same explicit wheel throughout so normal indexes cannot substitute another Torch.
        "$target/bin/python" -m pip install "${pip_args[@]}" -r "$requirements" "${wheel[0]}"
        echo "Installing Torch last: torch==$torch_version"
        "$target/bin/python" -m pip install "${pip_args[@]}" "${wheel[0]}"
    )
    "$target/bin/python" -m pip check

    "$target/bin/python" -B - "$CHANNEL" <<'PY'
import sys
import torch
import transformers
import swift

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

target="$ROOT/.venv/train"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Only Linux x86_64 is supported."
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python 3.12 not found: $PYTHON_BIN. Install it or set PYTHON_BIN to its path."
check_python "$PYTHON_BIN"
if [[ -e "$target" ]]; then
    [[ -x "$target/bin/python" ]] || die "Existing environment is incomplete: $target; move it aside before recreating it."
    check_python "$target/bin/python"
fi
select_torch
printf 'train CUDA wheel index: %s/%s/\n' "${TORCH_INDEX_BASE%/}" "$CHANNEL"
install_environment "$target" "$ROOT/requirements/train.txt"
printf '\ntrain environment: %s\nHF endpoint: %s\n' "$target" "$HF_ENDPOINT"
