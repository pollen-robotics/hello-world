pushd ~/dev/hello-world
    systemctl --user stop reachy_teleop.service

    status=$(systemctl --user is-active reachy_sdk_grpc.service)

    if [ "$status" != "active" ]; then
        systemctl --user restart reachy_sdk_grpc.service
        sleep 30
    fi

    python3 -m hello_world.hello
popd