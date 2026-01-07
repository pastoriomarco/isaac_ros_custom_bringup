# Jetson Remote Connection

This is a **from-scratch, reproducible, step-by-step** for **Ubuntu 24.04 laptop ↔ Jetson Orin (JetPack 6.x / Ubuntu 22.04) and Jetson Thor (JetPack 7.2 / Ubuntu 24.04)** to get an **interactive remote screen** (the device’s real `:0` desktop mirrored to the laptop) using **x11vnc + SSH tunnel + TigerVNC viewer**.

The document contains **only the required command sequence**, split into steps.
Where Orin and Thor differ, **both variants are explicitly shown**.
Where they are identical, **no platform is specified**.

---

## 0) Assumptions / goal

* The Jetson is reachable via SSH
* An **HDMI dummy plug** is attached to Thor or a **DisplayPort dummy plug** to Orin 
* You want to mirror/control the **real local desktop**, not a virtual VNC desktop

---

## 1) Laptop setup (Ubuntu 24.04)

### 1.1 Install VNC viewer and SSH client

**Laptop terminal (local):**

```bash
sudo apt update
sudo apt install -y tigervnc-viewer openssh-client
```

---

## 2) Jetson setup

### 2.1 Force Xorg and enable autologin

#### Jetson Orin/Thor (JetPack 6.x / 7.x / Ubuntu 22.04)

**Laptop terminal (local):**

Connect to the Jetson using its current username and IP

```bash
ssh <username>@<IP>
```

**Jetson (SSH):**

```bash
sudo nano /etc/gdm3/custom.conf
```

Ensure under `[daemon]`:

```ini
WaylandEnable=false
AutomaticLoginEnable=true
AutomaticLogin=<username>
```

```bash
sudo reboot
```

Reconnect after reboot:

```bash
ssh <username>@<IP>
```

---

### 2.2 Install x11vnc (both platforms)

**Jetson (SSH):**

```bash
sudo apt update
sudo apt install -y x11vnc
```

---

## 3) Start the remote desktop session

### 3.1 Quick start (recommended): use the scripts

From this folder on your laptop:

```bash
chmod +x ./connect_thor.sh ./connect_orin.sh
```

#### Thor

```bash
./connect_thor.sh
```

Override the default IP if needed:

```bash
./connect_thor.sh --ip 192.168.1.xx
```

#### Orin

```bash
./connect_orin.sh
```

Override the default IP if needed:

```bash
./connect_orin.sh --ip 192.168.1.xx
```

The script:

* opens a master SSH connection
* starts `x11vnc` on the Jetson
* sets up the SSH tunnel
* opens `vncviewer`

Stop it with `Ctrl+C` in the terminal running the script.

---

### 3.2 Manual startup (3 laptop terminals + viewer)

You will use:

* **Laptop Terminal A** → SSH tunnel (stays open)
* **Laptop Terminal B** → SSH into Jetson and run x11vnc (stays open)
* **Laptop Terminal C** → VNC viewer

---

#### 3.2.1 Laptop Terminal A: start SSH tunnel

**Laptop Terminal A (local):**

```bash
ssh -N -T -o ExitOnForwardFailure=yes -L 5906:127.0.0.1:5900 <username>@<IP>
```

Leave this terminal open.

---

#### 3.2.2 Laptop Terminal B: start x11vnc on the Jetson

**Laptop Terminal B (local):**

```bash
ssh <username>@<IP>
```

#### Jetson Orin

```bash
sudo -u $(whoami) env \
  DISPLAY=:0 \
  XAUTHORITY=~/.Xauthority \
  x11vnc -display :0 -localhost -forever -noxdamage -nopw -rfbport 5900
```

---

#### Jetson Thor

```bash
sudo -u $(whoami) env \
  DISPLAY=:0 \
  XAUTHORITY=/run/user/2002/gdm/Xauthority \
  XAUTHLOCALHOSTNAME=localhost \
  x11vnc -display :0 -localhost -forever -noxdamage -nopw -rfbport 5900
```

Leave this terminal open.

---

#### 3.2.3 Laptop Terminal C: connect with the VNC viewer

**Laptop Terminal C (local):**

```bash
vncviewer 127.0.0.1:5906
```

(If needed, `xtigervncviewer 127.0.0.1:5906` works as well.)

You should now see and control the Jetson’s desktop.

---

## 4) Set display resolution to 1080p (optional, both platforms)

**Jetson (SSH) while GUI is running:**

```bash
DISPLAY=:0 xrandr --output HDMI-0 --mode 1920x1080 --rate 60
```

---
