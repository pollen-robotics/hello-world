pushd ~/dev/hello-world
    status=$(systemctl --user is-active reachy_sdk_grpc.service)

    if [ "$status" != "active" ]; then
        systemctl --user restart reachy_sdk_grpc.service
        sleep 30
    fi

    python3 -m hello_world.hello
popd