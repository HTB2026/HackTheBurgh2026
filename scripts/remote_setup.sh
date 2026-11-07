#!/usr/bin/env bash
# Runs ON the server, streamed by scripts/deploy.sh, which first sends
# `read -r SUDO_PASS` and the password line, so SUDO_PASS is already set here.
set -euo pipefail
s() { printf '%s\n' "$SUDO_PASS" | sudo -S -p '' "$@"; }
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"
cd ~/earwitness

echo "--> system packages"
s apt-get update -qq
s apt-get install -y -qq git curl build-essential rsync ffmpeg tesseract-ocr tesseract-ocr-eng nginx docker.io >/dev/null

# ffmpeg must be >= 5.1 (-fps_mode). Ubuntu 22.04 ships 4.4: use a static build then.
if ! ffmpeg -hide_banner -h full 2>/dev/null | grep -q -- '-fps_mode'; then
  echo "--> installing static ffmpeg (system one is too old)"
  curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz
  tar -xf /tmp/ffmpeg.tar.xz -C /tmp
  s cp /tmp/ffmpeg-*-amd64-static/ffmpeg /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/
fi

echo "--> node 22"
node_ok() { command -v npm >/dev/null && node -v 2>/dev/null | grep -qE '^v(2[2-9]|[3-9][0-9])'; }

if ! node_ok; then
  curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource.sh
  s bash /tmp/nodesource.sh
  s apt-get install -y -qq nodejs
  hash -r
fi

if ! command -v npm >/dev/null; then
  echo "NodeSource gave no npm; falling back to Ubuntu's npm"
  s apt-get install -y -qq npm
  hash -r
fi

echo "node $(node -v), npm $(npm -v)"

echo "--> uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
hash -r
UV="$(command -v uv)"
echo "uv at $UV"

# .env must end with a newline before anything is appended.
touch .env
[[ -z "$(tail -c1 .env)" ]] || echo >> .env

echo "--> database"
if ! grep -q '^TIMESCALE_SERVICE_URL=' .env; then
  s systemctl enable --now docker >/dev/null
  if ! s docker ps -a --format '{{.Names}}' | grep -qx tsdb; then
    s docker run -d --name tsdb --restart unless-stopped -p 127.0.0.1:5432:5432 \
      -e POSTGRES_PASSWORD=earwitness timescale/timescaledb:latest-pg17 >/dev/null
  fi
  echo "TIMESCALE_SERVICE_URL=postgres://postgres:earwitness@127.0.0.1:5432/postgres" >> .env
fi

if s docker ps --format '{{.Names}}' 2>/dev/null | grep -qx tsdb; then
  for _ in $(seq 1 30); do s docker exec tsdb pg_isready -q && break; sleep 2; done
fi

echo "--> geo-sleuth"
mkdir -p ~/.cache/earwitness/vendor
[[ -d ~/.cache/earwitness/vendor/geo-sleuth ]] || \
  git clone -q --depth 1 https://github.com/Oldcircle/geo-sleuth ~/.cache/earwitness/vendor/geo-sleuth
sed -i '/^EARWITNESS_GEOSLEUTH=/d' .env
echo "EARWITNESS_GEOSLEUTH=$HOME/.cache/earwitness/vendor/geo-sleuth/skills/geo-sleuth" >> .env

echo "--> python packages"
uv sync -q

echo "--> web UI"
(cd web && npm install --no-audit --no-fund --loglevel=error && npm run build >/dev/null)

echo "--> migrate"
uv run earwitness db migrate

echo "--> service"
cat > /tmp/earwitness.service <<EOF
[Unit]
Description=Earwitness video forensics
After=network-online.target docker.service

[Service]
User=$USER
WorkingDirectory=$HOME/earwitness
Environment=HOME=$HOME
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$UV run earwitness serve --host 127.0.0.1 --port 8765
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
s cp /tmp/earwitness.service /etc/systemd/system/earwitness.service
s systemctl daemon-reload
s systemctl enable earwitness >/dev/null
s systemctl restart earwitness

echo "--> nginx"
cat > /tmp/earwitness.nginx <<'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    client_max_body_size 4G;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;

    location /ws/ {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_request_buffering off;
    }
}
EOF
s cp /tmp/earwitness.nginx /etc/nginx/sites-available/earwitness
s ln -sf /etc/nginx/sites-available/earwitness /etc/nginx/sites-enabled/earwitness
s rm -f /etc/nginx/sites-enabled/default
s nginx -t -q
s systemctl reload nginx

if s ufw status 2>/dev/null | grep -q "Status: active"; then
  s ufw allow 80/tcp >/dev/null
fi

echo "--> health check"
for _ in $(seq 1 30); do
  if curl -fs http://127.0.0.1/api/health >/dev/null; then
    curl -s http://127.0.0.1/api/health | head -c 120; echo
    exit 0
  fi
  sleep 2
done

echo "Service did not come up; last logs:"
s journalctl -u earwitness -n 40 --no-pager
exit 1
