tee app_reachy_idle.service <<EOF
[Unit]
Description=Reachy Idle service

[Service]
ExecStart=/usr/bin/bash $PWD/launch.bash

[Install]
WantedBy=default.target
EOF

mkdir -p $HOME/.config/systemd/user

mv app_reachy_idle.service $HOME/.config/systemd/user

echo ""
echo "app_reachy_idle.service is now setup."