#!/usr/bin/env bash

export PYTHONUNBUFFERED=TRUE
export PYTHONWARNINGS=once

# Calling the slicer_cli_web.cli_list_entrypoint always works, but we can skip
# an extra exec if we find the path directly; we can just dump xml if that is
# requested
POSSIBLE_PATH="$1/$1.py"
POSSIBLE_XML_PATH="$1/$1.xml"
if [[ -f "$POSSIBLE_PATH" && -f "$POSSIBLE_XML_PATH" && "$2" == "--xml" ]]; then
    cat "$POSSIBLE_XML_PATH"
else
    # Try to start a local version memcached, but fail gracefully if we can't.
    memcached -u root -d -m 1024 || true

    if [[ -f "$POSSIBLE_PATH" ]]; then
        python -W once::FutureWarning -u "$POSSIBLE_PATH" "${@:2}"
    else
        python -W once::FutureWarning -u -m slicer_cli_web.cli_list_entrypoint "$@"
    fi
fi
