import os
from pathlib import Path

import pytest

cliDir = Path(__file__).parent / '..' / 'histomicstk' / 'cli'
xmlList = {
    cli: cliDir / cli / f'{cli}.xml' for cli in os.listdir(cliDir)
    if (cliDir / cli).is_dir() and (cliDir / cli / f'{cli}.xml').is_file()
}


@pytest.mark.parametrize('cli', xmlList.keys())
def testXMLParsing(cli, caplog):
    import slicer_cli_web

    slicer_cli_web.CLIArgumentParser(xmlList[cli])
    for record in caplog.records:
        assert record.levelname not in {'WARNING', 'ERROR'}
