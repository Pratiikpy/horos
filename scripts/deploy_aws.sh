#!/bin/bash
# Configure the Horos host: systemd units, nginx, and the ledger.
#
# Two services, deliberately separate:
#
#   horos-api       the HTTP surface. Must never be down — OKX's reviewers probe unannounced, and
#                   "uptime at their test moment" is what failed most agents in live review.
#   horos-recorder  the forecast round, the scorer and the anchor job. Runs at nice 10 so that an
#                   incoming paid request always preempts inference rather than queuing behind it.
#                   This is the whole reason the two are split: on 2 vCPU, a 16-minute generation
#                   would otherwise make every API call during it look like a timeout.
#
# The recorder is the ONLY writer of the ledger, enforced by a pid lock file. Two writers appending
# to one hash chain would not lose data — they would produce a chain that fails verification
# permanently, which for this product is worse.
set -euo pipefail

APP=/opt/horos
PY=$APP/venv/bin/python

echo "==> writing systemd units"
sudo tee /etc/systemd/system/horos-api.service >/dev/null <<UNIT
[Unit]
Description=Horos ASP - x402 API (40 services)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP
EnvironmentFile=$APP/.env
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=2
ExecStart=$APP/venv/bin/uvicorn server:app --host 127.0.0.1 --port 8080 --workers 1 --timeout-keep-alive 65
Restart=always
RestartSec=3
# The reviewer's probe must never meet a dead socket.
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
UNIT

sudo tee /etc/systemd/system/horos-recorder.service >/dev/null <<UNIT
[Unit]
Description=Horos recorder - issues, scores and anchors forecasts
After=network-online.target horos-api.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP
EnvironmentFile=$APP/.env
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=2
# Inference yields to the API. On two shared cores this is what keeps paid calls responsive while a
# forecast round is running.
Nice=10
ExecStart=$APP/venv/bin/python -m scripts.recorder_daemon --device cpu
Restart=always
RestartSec=30
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
UNIT

echo "==> writing nginx site"
sudo tee /etc/nginx/sites-available/horos >/dev/null <<'NGINX'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # A forecast for an off-list symbol runs the model live. x402 declares a 300s ceiling, so the
    # proxy must not give up before the service does — a 504 here would be indistinguishable to the
    # caller from the service being broken.
    proxy_read_timeout 310s;
    proxy_send_timeout 310s;
    client_max_body_size 4m;

    # Payment headers must survive the hop. Losing either one hands a 402 to a caller who has
    # already paid. `PAYMENT-SIGNATURE` and `X-PAYMENT` are hyphenated so nginx keeps them by
    # default, but a client sending the underscore spelling would otherwise be silently dropped —
    # and this is a server-level directive, not a location-level one.
    underscores_in_headers on;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass_request_headers on;
    }
}
NGINX

sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/horos /etc/nginx/sites-enabled/horos
sudo nginx -t

echo "==> starting"
sudo systemctl daemon-reload
sudo systemctl enable --now horos-api horos-recorder
sudo systemctl reload nginx

sleep 6
echo "==> status"
systemctl is-active horos-api horos-recorder nginx || true
curl -fsS http://127.0.0.1:8080/health | head -c 400; echo
echo "==> done"
