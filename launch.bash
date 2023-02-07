pushd ~/dev/hello-world
    status=$(systemctl --user is-active reachy_sdk_server.service)

    if [ "$status" != "active" ]; then
        systemctl --user restart reachy_sdk_server.service
        sleep 30
    fi

    python3 -m hello_world.hello
popd