import os

from slicer_cli_web import ctk_cli_adjustment  # noqa
import slicer_cli_web.cli_list_entrypoint  # noqa I001


def main():
    slicer_cli_web.cli_list_entrypoint.CLIListEntrypoint(
        cwd=os.path.dirname(os.path.realpath(__file__)))
