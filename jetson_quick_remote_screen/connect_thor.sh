#!/usr/bin/env bash
set -euo pipefail

JETSON_USER="tndlux"
DEFAULT_JETSON_HOST="192.168.1.18"
JETSON_HOST="$DEFAULT_JETSON_HOST"

LOCAL_PORT="5906"
REMOTE_PORT="5900"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--ip <JETSON_IP>]

Defaults:
  --ip ${DEFAULT_JETSON_HOST}
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

cleanup() {
  echo "[*] Cleaning up..."

  if [ -S "$SOCK" ]; then
    # Stop remote x11vnc via the same master connection
    ssh -o BatchMode=yes -S "$SOCK" "${JETSON_USER}@${JETSON_HOST}" \
      "if [ -f '$REMOTE_PID_FILE' ]; then
          pid=\$(cat '$REMOTE_PID_FILE' 2>/dev/null || true);
          if [ -n \"\$pid\" ]; then kill \"\$pid\" >/dev/null 2>&1 || true; fi;
          rm -f '$REMOTE_PID_FILE' >/dev/null 2>&1 || true;
       fi" >/dev/null 2>&1 || true

    # Close master (also closes tunnel)
    ssh -o BatchMode=yes -S "$SOCK" -O exit "${JETSON_USER}@${JETSON_HOST}" >/dev/null 2>&1 || true
    rm -f "$SOCK" >/dev/null 2>&1 || true
  else
    rm -f "$SOCK" >/dev/null 2>&1 || true
  fi

  echo "[*] Done."
}
trap cleanup EXIT INT TERM

echo "[*] Opening master SSH connection (you should authenticate once)..."
ssh -M -S "$SOCK" -o ControlPersist=yes -fN "${JETSON_USER}@${JETSON_HOST}"

echo "[*] Starting remote x11vnc on Thor (${JETSON_HOST})..."
ssh -S "$SOCK" "${JETSON_USER}@${JETSON_HOST}" "bash -lc '
  set -e
  pkill -x x11vnc >/dev/null 2>&1 || true

  AUTH_ARG=\"\"
  RUN_AS_USER=\"${JETSON_USER}\"

  if [ -r \"/home/${JETSON_USER}/.Xauthority\" ]; then
    AUTH_ARG=\"-auth /home/${JETSON_USER}/.Xauthority\"
  elif [ -r \"/run/user/\$(id -u gdm)/gdm/Xauthority\" ]; then
    AUTH_ARG=\"-auth /run/user/\$(id -u gdm)/gdm/Xauthority\"
    RUN_AS_USER=\"root\"
  else
    AUTH_ARG=\"-auth guess\"
    RUN_AS_USER=\"root\"
  fi

  if [ \"\$RUN_AS_USER\" = \"root\" ]; then
    sudo env \
      DISPLAY=:0 \
      XAUTHLOCALHOSTNAME=localhost \
      nohup x11vnc \$AUTH_ARG -display :0 -localhost -forever -noxdamage -nopw -rfbport ${REMOTE_PORT} \
        >/tmp/x11vnc.log 2>&1 &
  else
    sudo -u ${JETSON_USER} env \
      DISPLAY=:0 \
      XAUTHORITY=/home/${JETSON_USER}/.Xauthority \
      nohup x11vnc \$AUTH_ARG -display :0 -localhost -forever -noxdamage -nopw -rfbport ${REMOTE_PORT} \
        >/tmp/x11vnc.log 2>&1 &
  fi

  echo \$! > ${REMOTE_PID_FILE}
  sleep 0.5
  pid=\$(cat ${REMOTE_PID_FILE})
  if ! kill -0 \"\$pid\" >/dev/null 2>&1; then
    echo \"[remote] x11vnc failed to stay up (pid=\$pid)\" >&2
    tail -n 40 /tmp/x11vnc.log >&2 || true
    exit 1
  fi
  echo \"[remote] x11vnc pid=\$pid\"
'"

echo "[*] Creating tunnel on the same master connection..."
ssh -S "$SOCK" -fN -T \
  -o ExitOnForwardFailure=yes \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${JETSON_USER}@${JETSON_HOST}"

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
