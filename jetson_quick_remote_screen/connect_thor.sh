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
  echo "[*] Forcing 1080p on the Jetson (DISPLAY=:0, HDMI-0)..."
  ssh "${ssh_master_opts[@]}" "${ssh_no_prompt_opts[@]}" "${JETSON_USER}@${JETSON_HOST}" \
    "bash -lc 'DISPLAY=:0 xrandr --output HDMI-0 --mode 1920x1080 --rate 60'" >/dev/null 2>&1 || true
  sleep 1
fi

echo "[*] Starting remote x11vnc on Thor (${JETSON_HOST})..."
ssh "${ssh_master_opts[@]}" "${ssh_no_prompt_opts[@]}" "${JETSON_USER}@${JETSON_HOST}" "bash -lc '
  set -e
  pkill -x x11vnc >/dev/null 2>&1 || true

  AUTH_ARG=\"\"
  RUN_AS_USER=\"${JETSON_USER}\"
  XAUTH_PATH=\"\"

  if [ -r \"/home/${JETSON_USER}/.Xauthority\" ]; then
    XAUTH_PATH=\"/home/${JETSON_USER}/.Xauthority\"
    AUTH_ARG=\"-auth \${XAUTH_PATH}\"
  else
    # Prefer the active seat0 GDM Xauthority, which is common on Thor.
    seat_uid=\$(loginctl list-sessions --no-legend 2>/dev/null | while read -r session uid user seat rest; do
      if [ \"\$seat\" = \"seat0\" ]; then
        echo \"\$uid\"
        break
      fi
    done)
    if [ -n \"\$seat_uid\" ] && [ -r \"/run/user/\${seat_uid}/gdm/Xauthority\" ]; then
      XAUTH_PATH=\"/run/user/\${seat_uid}/gdm/Xauthority\"
      AUTH_ARG=\"-auth \${XAUTH_PATH}\"
      RUN_AS_USER=\"root\"
    elif [ -r \"/run/user/\$(id -u gdm)/gdm/Xauthority\" ]; then
      XAUTH_PATH=\"/run/user/\$(id -u gdm)/gdm/Xauthority\"
      AUTH_ARG=\"-auth \${XAUTH_PATH}\"
      RUN_AS_USER=\"root\"
    else
      AUTH_ARG=\"-auth guess\"
      RUN_AS_USER=\"root\"
    fi
  fi

	  if [ \"\$RUN_AS_USER\" = \"root\" ]; then
	    sudo bash -lc \"env DISPLAY=:0 XAUTHLOCALHOSTNAME=localhost nohup x11vnc \$AUTH_ARG -display :0 -localhost -forever -noxdamage -nopw -rfbport ${REMOTE_PORT} >/tmp/x11vnc.log 2>&1 & echo \\\$! > ${REMOTE_PID_FILE}\"
	  else
	    env \
	      DISPLAY=:0 \
	      XAUTHORITY=\"\${XAUTH_PATH}\" \
      nohup x11vnc \$AUTH_ARG -display :0 -localhost -forever -noxdamage -nopw -rfbport ${REMOTE_PORT} \
        >/tmp/x11vnc.log 2>&1 &
    echo \$! > ${REMOTE_PID_FILE}
  fi
  sleep 0.5
  pid=\$(cat ${REMOTE_PID_FILE})
  if ! kill -0 \"\$pid\" >/dev/null 2>&1; then
    if ps -p \"\$pid\" >/dev/null 2>&1; then
      echo \"[remote] x11vnc is running (pid=\$pid) but owned by another user\"
    else
      echo \"[remote] x11vnc failed to stay up (pid=\$pid)\" >&2
      tail -n 40 /tmp/x11vnc.log >&2 || true
      exit 1
    fi
  fi
  echo \"[remote] x11vnc pid=\$pid\"
'"

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
