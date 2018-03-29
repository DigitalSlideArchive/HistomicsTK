#!/usr/bin/env bash

if type nvidia-smi >/dev/null 2>/dev/null;
then
    source /venv-gpu/bin/activate
    echo "NOTE: GPU available" >&2
fi

# Try to start a local version memcached, but fail gracefully if we can't.
memcached -u root -d -m 1024 || true

python /build/slicer_cli_web/server/cli_list_entrypoint.py "$@"
