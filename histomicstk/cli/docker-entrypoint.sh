#!/usr/bin/env bash

# Try to start a local version memcached, but fail gracefully if we can't.
memcached -u root -d -m 1024 || true

# Calling the slicer_cli_web.cli_list_entrypoint always works, but we can skip
# an extra exec if we find the path directly
POSSIBLE_PATH="$1/$1.py"
if [[ -f "$POSSIBLE_PATH" ]]; then
    python -u "$POSSIBLE_PATH" "${@:2}"
else
    python -u -m slicer_cli_web.cli_list_entrypoint "$@"
fi
