import os

import slicer_cli_web.cli_list_entrypoint  # noqa I001
from slicer_cli_web import ctk_cli_adjustment  # noqa


def main():
    slicer_cli_web.cli_list_entrypoint.CLIListEntrypoint(
        cwd=os.path.dirname(os.path.realpath(__file__)))
