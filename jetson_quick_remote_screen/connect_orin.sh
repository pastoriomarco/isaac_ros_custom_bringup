#!/usr/bin/env bash
set -euo pipefail

# Fresh Orin flash notes:
# 1. Use the USB-C port next to the 40-pin header. The Jetson should be
#    reachable over USB at 192.168.55.1 after first boot.
# 2. If SSH complains about a changed host key after reflashing, run on host:
#      ssh-keygen -R 192.168.55.1
# 3. If the Jetson has no internet yet, connect Wi-Fi from serial/SSH:
#      sudo nmcli device wifi connect "SSID" password "PASSWORD" ifname wlP1p1s0
# 4. Install x11vnc on the Jetson before using this script:
#      sudo apt update && sudo apt install -y x11vnc
# 5. If another Jetson remote screen is already using local 5906, run:
#      ./connect_orin.sh --ip 192.168.55.1 --local-port 5907
# 6. On a fresh graphical boot, the first run may show the GDM login screen.
#    Log in there; the VNC viewer can disconnect because GDM replaces the
#    greeter session with the user's desktop. Run this script a second time to
#    attach to the logged-in desktop.

JETSON_USER="tndlux"
DEFAULT_JETSON_HOST="192.168.1.17"
JETSON_HOST="$DEFAULT_JETSON_HOST"

LOCAL_PORT="5906"
REMOTE_PORT="5900"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--ip <JETSON_IP>] [--local-port <PORT>] [--remote-port <PORT>]

Defaults:
  --ip ${DEFAULT_JETSON_HOST}
  --local-port ${LOCAL_PORT}
  --remote-port ${REMOTE_PORT}
EOF
}

validate_port() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ] || [ "$value" -gt 65535 ]; then
    echo "error: ${name} must be a TCP port from 1 to 65535" >&2
    exit 2
  fi
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
    --local-port)
      shift
      if [ $# -eq 0 ]; then
        echo "error: --local-port requires a value" >&2
        usage >&2
        exit 2
      fi
      LOCAL_PORT="$1"
      shift
      ;;
    --local-port=*)
      LOCAL_PORT="${1#--local-port=}"
      shift
      ;;
    --remote-port)
      shift
      if [ $# -eq 0 ]; then
        echo "error: --remote-port requires a value" >&2
        usage >&2
        exit 2
      fi
      REMOTE_PORT="$1"
      shift
      ;;
    --remote-port=*)
      REMOTE_PORT="${1#--remote-port=}"
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
validate_port "--local-port" "$LOCAL_PORT"
validate_port "--remote-port" "$REMOTE_PORT"

SOCK="/tmp/vnc-${JETSON_HOST}-${LOCAL_PORT}.sock"
REMOTE_PID_FILE="/tmp/x11vnc-${JETSON_USER}-${REMOTE_PORT}.pid"
REMOTE_ROOT_PID_FILE="/tmp/x11vnc-${JETSON_USER}-${REMOTE_PORT}-root.pid"
REMOTE_USER_LOG_FILE="/tmp/x11vnc-${JETSON_USER}-${REMOTE_PORT}-user.log"
REMOTE_ROOT_LOG_FILE="/tmp/x11vnc-${JETSON_USER}-${REMOTE_PORT}-root.log"

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
      "for pid_file in '$REMOTE_PID_FILE' '$REMOTE_ROOT_PID_FILE'; do
        if [ -f \"\$pid_file\" ]; then
          pid=\$(cat \"\$pid_file\" 2>/dev/null || true);
          if [ -n \"\$pid\" ]; then
            kill \"\$pid\" >/dev/null 2>&1 || sudo -n kill \"\$pid\" >/dev/null 2>&1 || true;
          fi;
          rm -f \"\$pid_file\" >/dev/null 2>&1 || sudo -n rm -f \"\$pid_file\" >/dev/null 2>&1 || true;
        fi;
       done
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

echo "[*] Starting remote x11vnc on Orin (${JETSON_HOST})..."
if ! ssh "${ssh_master_opts[@]}" "${ssh_no_prompt_opts[@]}" "${JETSON_USER}@${JETSON_HOST}" "bash -lc '
  set -e
  pkill -x x11vnc >/dev/null 2>&1 || true
  rm -f ${REMOTE_USER_LOG_FILE} >/dev/null 2>&1 || true

  # A freshly logged-in GNOME session may use /run/user/<uid>/gdm/Xauthority
  # instead of ~/.Xauthority. Try both before falling back to sudo below.
  user_uid=\$(id -u)
  xauth_path=\"\"
  for candidate in \"/run/user/\${user_uid}/gdm/Xauthority\" \"/home/${JETSON_USER}/.Xauthority\"; do
    if [ -f \"\$candidate\" ] && [ -r \"\$candidate\" ]; then
      xauth_path=\"\$candidate\"
      break
    fi
  done

  if [ -z \"\$xauth_path\" ]; then
    echo \"[remote] no readable Xauthority file found for ${JETSON_USER}\" >&2
    exit 1
  fi

  env \
    XAUTHORITY=\"\$xauth_path\" \
    nohup x11vnc -auth \"\$xauth_path\" -find -localhost -forever -noxdamage -nopw -rfbport ${REMOTE_PORT} \
      >${REMOTE_USER_LOG_FILE} 2>&1 &

  echo \$! > ${REMOTE_PID_FILE}
  sleep 1
  pid=\$(cat ${REMOTE_PID_FILE})
  if ! kill -0 \"\$pid\" >/dev/null 2>&1; then
    echo \"[remote] x11vnc failed to stay up (pid=\$pid)\" >&2
    tail -n 80 ${REMOTE_USER_LOG_FILE} >&2 || true
    exit 1
  fi
  echo \"[remote] x11vnc pid=\$pid\"
'"; then
  echo "[*] User Xauthority did not work; trying sudo x11vnc with the active Xorg auth file..."
  ssh -t "${ssh_master_opts[@]}" "${JETSON_USER}@${JETSON_HOST}" "sudo bash -lc '
    set -e
    pkill -x x11vnc >/dev/null 2>&1 || true
    rm -f ${REMOTE_PID_FILE} ${REMOTE_ROOT_PID_FILE} >/dev/null 2>&1 || true
    rm -f ${REMOTE_ROOT_LOG_FILE} >/dev/null 2>&1 || true
    touch ${REMOTE_ROOT_LOG_FILE}
    chmod 644 ${REMOTE_ROOT_LOG_FILE}

    # If the screen is still owned by GDM, root must read the Xorg -auth file.
    # Parse it from the active Xorg command line, then let x11vnc -find choose
    # the display instead of hardcoding :0.
    auth_path=\"\"
    prev=\"\"
    while IFS= read -r word; do
      if [ \"\$prev\" = \"-auth\" ]; then
        auth_path=\"\$word\"
        break
      fi
      prev=\"\$word\"
    done < <(ps -eo args | tr \" \" \"\\n\")
    auth_args=(-auth guess)
    if [ -n \"\$auth_path\" ] && [ -r \"\$auth_path\" ]; then
      auth_args=(-auth \"\$auth_path\")
      echo \"[remote] using Xauthority: \$auth_path\"
    else
      echo \"[remote] using x11vnc -auth guess\"
    fi

    env XAUTHLOCALHOSTNAME=localhost \
      nohup x11vnc -env FD_XDM=1 \"\${auth_args[@]}\" -find -localhost -forever -noxdamage -nopw -rfbport ${REMOTE_PORT} \
        >${REMOTE_ROOT_LOG_FILE} 2>&1 &

    echo \$! > ${REMOTE_ROOT_PID_FILE}
    sleep 1
    pid=\$(cat ${REMOTE_ROOT_PID_FILE})
    if ! kill -0 \"\$pid\" >/dev/null 2>&1; then
      echo \"[remote] sudo x11vnc failed to stay up (pid=\$pid)\" >&2
      tail -n 80 ${REMOTE_ROOT_LOG_FILE} >&2 || true
      exit 1
    fi
    echo \"[remote] sudo x11vnc pid=\$pid\"
  '"
fi

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
  echo "[!] VNC server not ready after 20s; remote x11vnc log follows:" >&2
  ssh "${ssh_master_opts[@]}" "${ssh_no_prompt_opts[@]}" "${JETSON_USER}@${JETSON_HOST}" \
    "tail -n 120 ${REMOTE_USER_LOG_FILE} ${REMOTE_ROOT_LOG_FILE} 2>/dev/null" >&2 || true
  exit 1
fi

echo "[*] Opening VNC viewer..."
vncviewer "127.0.0.1:${LOCAL_PORT}"
