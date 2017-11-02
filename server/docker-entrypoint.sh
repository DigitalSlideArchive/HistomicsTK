#!/usr/bin/env bash

if type nvidia-smi >/dev/null 2>/dev/null;
then
    source /venv-gpu/bin/activate
fi

python /build/slicer_cli_web/server/cli_list_entrypoint.py "$@"