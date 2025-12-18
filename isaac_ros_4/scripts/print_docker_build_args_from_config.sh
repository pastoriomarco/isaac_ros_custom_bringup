#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: print_docker_build_args_from_config.sh

Prints `docker build` flags (to stdout) that toggle optional packages in `Dockerfile.isaac_manipulation`
based on the components config YAML. Also prints per-component sub-params as additional build args if present.

Example:
  docker build -f Dockerfile.isaac_manipulation \
    $(./scripts/print_docker_build_args_from_config.sh) \
    .

Reads config from:
  ../config/isaac_manipulation_assets.yaml (relative to this script)
EOF
}

default_config_from_script() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "${script_dir}/../config/isaac_manipulation_assets.yaml"
}

resolve_config_path() {
  default_config_from_script
}

component_enabled() {
  local cfg_path="$1"
  local component="$2"
  python3 - "${cfg_path}" "${component}" <<'PY'
import sys

cfg_path = sys.argv[1]
component = sys.argv[2]

def parse_scalar(raw: str):
    raw = raw.strip()
    if raw in ("true", "True"):
        return True
    if raw in ("false", "False"):
        return False
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        if raw.startswith("0") and raw != "0" and raw.isdigit():
            return raw
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Minimal YAML subset parser: supports nested mappings with 2-space indentation and scalar values.
    root = {}
    stack = [(0, root)]
    for original in text.splitlines():
        line = original.rstrip("\n")
        stripped = line.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].rstrip()
            if not stripped:
                continue
        indent = len(line) - len(stripped)
        if indent % 2 != 0:
            raise ValueError(f"unsupported indentation (expected multiples of 2): {original!r}")
        if ":" not in stripped:
            raise ValueError(f"expected 'key: value' mapping entry: {original!r}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation: {original!r}")
        cur = stack[-1][1]
        if rest == "":
            nxt = {}
            cur[key] = nxt
            stack.append((indent + 2, nxt))
        else:
            cur[key] = parse_scalar(rest)
    return root

try:
    cfg = load_cfg(cfg_path)
except Exception:
    print("1")
    raise SystemExit(0)

components = cfg.get("components", {})
value = components.get(component, False)

if isinstance(value, bool):
    enabled = value
elif isinstance(value, dict):
    enabled = bool(value.get("enabled", True))
else:
    enabled = True

print("1" if enabled else "0")
PY
}

print_component_params_as_build_args() {
  local cfg_path="$1"
  python3 - "${cfg_path}" <<'PY'
import re
import sys

cfg_path = sys.argv[1]

try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Minimal YAML subset parser: supports nested mappings with 2-space indentation and scalar values.
    root = {}
    stack = [(0, root)]

    def parse_scalar(raw: str):
        raw = raw.strip()
        if raw in ("true", "True"):
            return True
        if raw in ("false", "False"):
            return False
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            return raw[1:-1]
        try:
            if raw.startswith("0") and raw != "0" and raw.isdigit():
                return raw
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw

    for original in text.splitlines():
        line = original.rstrip("\n")
        stripped = line.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].rstrip()
            if not stripped:
                continue
        indent = len(line) - len(stripped)
        if indent % 2 != 0:
            raise ValueError(f"unsupported indentation (expected multiples of 2): {original!r}")
        if ":" not in stripped:
            raise ValueError(f"expected 'key: value' mapping entry: {original!r}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation: {original!r}")
        cur = stack[-1][1]
        if rest == "":
            nxt = {}
            cur[key] = nxt
            stack.append((indent + 2, nxt))
        else:
            cur[key] = parse_scalar(rest)

    cfg = root
except Exception:
    raise SystemExit(0)

components = cfg.get("components", {})
if not isinstance(components, dict):
    raise SystemExit(0)

def to_env_fragment(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value.upper()

def to_build_arg_value(value) -> str | None:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return None

lines: list[str] = []
for component_name, component_cfg in components.items():
    if not isinstance(component_cfg, dict):
        continue
    for key, value in component_cfg.items():
        if key == "enabled":
            continue
        rendered = to_build_arg_value(value)
        if rendered is None:
            continue
        arg_name = f"ISAAC_MANIPULATION_{to_env_fragment(component_name)}_{to_env_fragment(key)}"
        lines.append(f"--build-arg {arg_name}={rendered}")

for line in sorted(lines):
    print(line)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

CFG_PATH="$(resolve_config_path)"
if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: expected config file not found: ${CFG_PATH}" >&2
  exit 1
fi

enabled_or_zero() {
  local component="$1"
  if [[ "$(component_enabled "${CFG_PATH}" "${component}")" == "1" ]]; then
    echo 1
  else
    echo 0
  fi
}

cat <<EOF
--build-arg ISAAC_MANIPULATION_ENABLE_ESS=$(enabled_or_zero ess)
--build-arg ISAAC_MANIPULATION_ENABLE_FOUNDATIONSTEREO=$(enabled_or_zero foundationstereo)
--build-arg ISAAC_MANIPULATION_ENABLE_FOUNDATIONPOSE=$(enabled_or_zero foundationpose)
--build-arg ISAAC_MANIPULATION_ENABLE_RTDETR=$(enabled_or_zero rtdetr)
--build-arg ISAAC_MANIPULATION_ENABLE_GROUNDING_DINO=$(enabled_or_zero grounding_dino)
--build-arg ISAAC_MANIPULATION_ENABLE_SEGMENT_ANYTHING=$(enabled_or_zero segment_anything)
--build-arg ISAAC_MANIPULATION_ENABLE_DOPE=$(enabled_or_zero dope)
EOF

print_component_params_as_build_args "${CFG_PATH}"
