#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
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
    local cuda_version major minor cuda_code

    if [[ -n "${TORCH_CHANNEL:-}" ]]; then
        CHANNEL="$TORCH_CHANNEL"
    elif ! command -v nvidia-smi >/dev/null 2>&1 || \
         ! nvidia-smi -L 2>/dev/null | grep -q '^GPU [0-9]'; then
        [[ "$ALLOW_CPU" == "1" ]] || die "NVIDIA GPU/driver unavailable; set ALLOW_CPU=1 for CPU-only installation."
        CHANNEL=cpu
    else
        cuda_version="$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version:[[:space:]]*([0-9]+\.[0-9]+).*/\1/p' | head -n 1)" || true
        [[ -n "$cuda_version" ]] || die "Cannot detect CUDA capability from nvidia-smi; set TORCH_CHANNEL explicitly."
        major="${cuda_version%%.*}"
        minor="${cuda_version#*.}"
        cuda_code=$((10#$major * 100 + 10#$minor))

        if (( cuda_code >= 1208 )); then CHANNEL=cu128
        elif (( cuda_code >= 1206 )); then CHANNEL=cu126
        elif (( cuda_code >= 1204 )); then CHANNEL=cu124
        elif (( cuda_code >= 1201 )); then CHANNEL=cu121
        elif (( cuda_code >= 1108 )); then CHANNEL=cu118
        else die "Driver CUDA capability $cuda_version is too old for this script."
        fi
        echo "NVIDIA driver CUDA capability: $cuda_version"
    fi

    case "$CHANNEL" in
        cu128|cu126|cu124|cu121|cu118|cpu) ;;
        *) die "Unsupported TORCH_CHANNEL=$CHANNEL" ;;
    esac
}

install_environment() {
    local target="$1" requirements="$2"
    local torch_version
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

    "$target/bin/python" -m pip install "${pip_args[@]}" -r "$requirements"
    # Keep the Torch release chosen by requirements (including vLLM), changing only its CUDA build.
    torch_version="$("$target/bin/python" -c 'from importlib.metadata import version; print(version("torch").split("+")[0])')"
    echo "Installing Torch last: torch==$torch_version+$CHANNEL"
    (
        wheel_dir="$(mktemp -d)"
        trap 'rm -rf -- "$wheel_dir"' EXIT
        # Fetch only the Torch wheel from the CUDA source; resolve dependencies from the normal source.
        "$target/bin/python" -m pip download "${torch_args[@]}" \
            --no-deps --only-binary=:all: --dest "$wheel_dir" "torch==$torch_version+$CHANNEL" \
            || die "No Torch $torch_version build for $CHANNEL; this requirements/CUDA combination is unavailable."
        "$target/bin/python" -m pip install "${pip_args[@]}" "$wheel_dir"/torch-*.whl
    )
    "$target/bin/python" -m pip check

    "$target/bin/python" -B - "$CHANNEL" "${target##*/}" <<'PY'
import sys
import torch
import transformers
import swift

channel, role = sys.argv[1:]
print(f"Python: {sys.executable}; transformers={transformers.__version__}; torch={torch.__version__}")
if role == "evaluation":
    import lm_eval
    from vllm import LLM

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

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Only Linux x86_64 is supported."
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python 3.12 not found: $PYTHON_BIN. Install it or set PYTHON_BIN to its path."
check_python "$PYTHON_BIN"
for target in "$ROOT/.venv/train" "$ROOT/.venv/evaluation"; do
    if [[ -e "$target" ]]; then
        [[ -x "$target/bin/python" ]] || die "Existing environment is incomplete: $target; move it aside before recreating it."
        check_python "$target/bin/python"
    fi
done
select_torch
printf 'Selected CUDA wheel index: %s/%s/\n' "${TORCH_INDEX_BASE%/}" "$CHANNEL"

install_environment "$ROOT/.venv/train" "$ROOT/requirements/train.txt"
install_environment "$ROOT/.venv/evaluation" "$ROOT/requirements/evaluation.txt"

printf '\nTraining environment:   %s\nEvaluation environment: %s\nHF endpoint:            %s\n' \
    "$ROOT/.venv/train" "$ROOT/.venv/evaluation" "$HF_ENDPOINT"
