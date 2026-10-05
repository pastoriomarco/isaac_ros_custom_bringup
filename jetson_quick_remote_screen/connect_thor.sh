#!/usr/bin/env bash
set -euo pipefail

JETSON_USER="tndlux"
DEFAULT_JETSON_HOST="192.168.1.18"
JETSON_HOST="$DEFAULT_JETSON_HOST"

LOCAL_PORT="5906"
REMOTE_PORT="5900"
ENABLE_1080P="no"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--ip <JETSON_IP>]

Defaults:
  --ip ${DEFAULT_JETSON_HOST}
  --1080p (disabled)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --ip)
      shift
      if [ $# -eq 0 ]; then
        echo "error: --ip requires a value" >&2
        usage >&2
        exit 2
      fi
      JETSON_HOST="$1"
      shift
      ;;
    --ip=*)
      JETSON_HOST="${1#--ip=}"
      shift
      ;;
    --1080p)
      ENABLE_1080P="yes"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$JETSON_HOST" ]; then
  echo "error: --ip cannot be empty" >&2
  exit 2
fi

SOCK="/tmp/vnc-${JETSON_HOST}.sock"
REMOTE_PID_FILE="/tmp/x11vnc-${JETSON_USER}.pid"

ssh_master_opts=(
  -S "$SOCK"
  -o ControlMaster=auto
  -o ControlPersist=yes
  -o LogLevel=ERROR
)

ssh_no_prompt_opts=(
  -o BatchMode=yes
  -o ConnectTimeout=2
  -o ConnectionAttempts=1
)

timeout_cmd=()
if command -v timeout >/dev/null 2>&1; then
  timeout_cmd=(timeout 3)
fi

ssh_master_is_alive() {
  "${timeout_cmd[@]}" ssh "${ssh_master_opts[@]}" "${ssh_no_prompt_opts[@]}" -O check "${JETSON_USER}@${JETSON_HOST}" >/dev/null 2>&1
}

cleanup() {
  echo "[*] Cleaning up..."

  if ssh_master_is_alive; then
    # Stop remote x11vnc via the same master connection
    "${timeout_cmd[@]}" ssh "${ssh_master_opts[@]}" "${ssh_no_prompt_opts[@]}" "${JETSON_USER}@${JETSON_HOST}" \
      "if [ -f '$REMOTE_PID_FILE' ]; then
          pid=\$(cat '$REMOTE_PID_FILE' 2>/dev/null || true);
          if [ -n \"\$pid\" ]; then
            kill \"\$pid\" >/dev/null 2>&1 || sudo -n kill \"\$pid\" >/dev/null 2>&1 || true;
          fi;
          rm -f '$REMOTE_PID_FILE' >/dev/null 2>&1 || true;
       fi
       pkill -x x11vnc >/dev/null 2>&1 || sudo -n pkill -x x11vnc >/dev/null 2>&1 || true" >/dev/null 2>&1 || true

    # Close master (also closes tunnel)
    "${timeout_cmd[@]}" ssh "${ssh_master_opts[@]}" "${ssh_no_prompt_opts[@]}" -O exit "${JETSON_USER}@${JETSON_HOST}" >/dev/null 2>&1 || true
  fi

  rm -f "$SOCK" >/dev/null 2>&1 || true

  echo "[*] Done."
}
trap cleanup EXIT INT TERM

echo "[*] Opening master SSH connection (you should authenticate once)..."
if ssh_master_is_alive; then
  echo "[*] Reusing existing SSH master at ${SOCK}..."
else
  rm -f "$SOCK" >/dev/null 2>&1 || true
  ssh -M -S "$SOCK" \
    -o ControlPersist=yes \
    -o ServerAliveInterval=2 \
    -o ServerAliveCountMax=1 \
    -o ExitOnForwardFailure=yes \
    -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    -fN "${JETSON_USER}@${JETSON_HOST}"
fi

if [ "$ENABLE_1080P" = "yes" ]; then
  echo "[*] Forcing 1080p on the Jetson (HDMI-0) whenever x11vnc attaches..."
fi

echo "[*] Starting remote x11vnc on Thor (${JETSON_HOST})..."
# Runs as root: at the GDM login screen the X authority file belongs to gdm and
# tndlux cannot read it. A follower loop re-attaches x11vnc to the X server on
# the active VT, so the view survives the login screen -> user session hand-off
# (TigerVNC offers to reconnect when the old X server goes away).
ssh "${ssh_master_opts[@]}" "${ssh_no_prompt_opts[@]}" "${JETSON_USER}@${JETSON_HOST}" \
  bash -s -- "$REMOTE_PORT" "$REMOTE_PID_FILE" "$ENABLE_1080P" <<'REMOTE'
set -eu
port="$1"; pid_file="$2"; force_1080p="$3"

if ! sudo -n true 2>/dev/null; then
  echo "[remote] passwordless sudo is required to attach to the GDM display" >&2
  exit 1
fi

old_pid=$(cat "$pid_file" 2>/dev/null || true)
if [ -n "$old_pid" ] && grep -qa x11vnc-follow "/proc/$old_pid/cmdline" 2>/dev/null; then
  sudo -n kill "$old_pid" >/dev/null 2>&1 || true
fi
sudo -n pkill -x x11vnc >/dev/null 2>&1 || true

sudo -n env PORT="$port" PID_FILE="$pid_file" FORCE_1080P="$force_1080p" \
  nohup bash -c '
    # x11vnc-follow
    exec >>/tmp/x11vnc.log 2>&1
    echo $$ > "$PID_FILE"
    find_x() {
      vt=$(cat /sys/class/tty/tty0/active); vt=${vt#tty}
      for pid in $(pgrep -x Xorg); do
        args=" $(tr "\0" " " </proc/$pid/cmdline)"
        case "$args" in *" vt$vt "*) ;; *) continue ;; esac
        auth=$(printf "%s\n" "$args" | sed -n "s/.* -auth \([^ ]*\).*/\1/p")
        sock=$(ss -xlpn | grep "pid=$pid," | grep -o "/tmp/.X11-unix/X[0-9]*" | head -n1)
        if [ -n "$auth" ] && [ -n "$sock" ]; then
          echo ":${sock##*X} $auth"
          return 0
        fi
      done
      return 1
    }
    while true; do
      if x=$(find_x); then
        set -- $x
        echo "[follow] attaching to display $1 (auth $2)"
        if [ "$FORCE_1080P" = yes ]; then
          DISPLAY="$1" XAUTHORITY="$2" xrandr --output HDMI-0 --mode 1920x1080 --rate 60 || true
        fi
        env DISPLAY="$1" XAUTHLOCALHOSTNAME=localhost \
          x11vnc -display "$1" -auth "$2" -localhost -forever -noxdamage -nopw -rfbport "$PORT" || true
      fi
      sleep 1
    done
  ' >/dev/null 2>&1 </dev/null &

for _ in $(seq 1 20); do
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    echo "[remote] x11vnc follower pid=$(cat "$pid_file" 2>/dev/null) listening on $port"
    exit 0
  fi
  sleep 0.5
done
echo "[remote] x11vnc did not start listening on port $port" >&2
tail -n 40 /tmp/x11vnc.log >&2 || true
exit 1
REMOTE

echo "[*] Waiting for VNC server to be ready..."
ready=""
deadline=$((SECONDS + 20))
while [ "$SECONDS" -lt "$deadline" ]; do
  if exec 3<>/dev/tcp/127.0.0.1/"${LOCAL_PORT}" 2>/dev/null; then
    if read -r -n 3 -t 1 banner <&3; then
      if [ "$banner" = "RFB" ]; then
        ready="yes"
      fi
    fi
    exec 3<&- 3>&-
  fi
  if [ -n "$ready" ]; then
    break
  fi
  sleep 0.3
done
if [ -z "$ready" ]; then
  echo "[!] VNC server not ready after 20s; opening viewer anyway..."
fi

echo "[*] Opening VNC viewer..."
vncviewer "127.0.0.1:${LOCAL_PORT}"

