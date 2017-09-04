import os
import sys
import json
import argparse
import subprocess
import textwrap as _textwrap

try:
    from girder import logger
except ImportError:
    import logging as logger


class _MultilineHelpFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        paragraphs = text.split('|n')
        multiline_text = ''
        for paragraph in paragraphs:
            formatted_paragraph = '\n' + _textwrap.fill(
                paragraph, width,
                initial_indent=indent,
                subsequent_indent=indent) + '\n'
            multiline_text += formatted_paragraph
        return multiline_text


def _make_print_cli_list_spec_action(cli_list_spec_file):

    with open(cli_list_spec_file) as f:
        str_cli_list_spec = f.read()

    class _PrintCLIListSpecAction(argparse.Action):

        def __init__(self,
                     option_strings,
                     dest=argparse.SUPPRESS,
                     default=argparse.SUPPRESS,
                     help=None):
            super(_PrintCLIListSpecAction, self).__init__(
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                help=help)

        def __call__(self, parser, namespace, values, option_string=None):
            print str_cli_list_spec
            parser.exit()

    return _PrintCLIListSpecAction


def CLIListEntrypoint(cli_list_spec_file=None):

    if cli_list_spec_file is None:
        cli_list_spec_file = os.path.join(os.getcwd(), 'slicer_cli_list.json')

    # Parse CLI List spec
    with open(cli_list_spec_file) as f:
        cli_list_spec = json.load(f)

    # create command-line argument parser
    cmdparser = argparse.ArgumentParser(
        formatter_class=_MultilineHelpFormatter
    )

    # add --cli_list
    cmdparser.add_argument(
        '--list_cli',
        action=_make_print_cli_list_spec_action(cli_list_spec_file),
        help='Prints the json file containing the list of CLIs present'
    )

    # add cl-rel-path argument
    cmdparser.add_argument("cli",
                           help="CLI to run",
                           metavar='<cli>',
                           choices=cli_list_spec.keys())

    args = cmdparser.parse_args(sys.argv[1:2])

    args.cli = os.path.normpath(args.cli)

    if cli_list_spec[args.cli]['type'] == 'python':

        script_file = os.path.join(args.cli,
                                   os.path.basename(args.cli) + '.py')

        # python <cli-rel-path>/<cli-name>.py [<args>]
        subprocess.call([sys.executable, script_file] + sys.argv[2:])

    elif cli_list_spec[args.cli]['type'] == 'cxx':

        script_file = os.path.join('.', args.cli, os.path.basename(args.cli))

        # ./<cli-rel-path>/<cli-name> [<args>]
        subprocess.call([script_file] + sys.argv[2:])

    else:
        logger.exception('CLIs of type %s are not supported',
                         cli_list_spec[args.cli]['type'])
        raise Exception(
            'CLIs of type %s are not supported',
            cli_list_spec[args.cli]['type']
        )


if __name__ == "__main__":
    CLIListEntrypoint()
