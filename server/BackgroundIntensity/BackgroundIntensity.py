from ctk_cli import CLIArgumentParser
from dask.distributed import Client

from histomicstk.preprocessing.color_normalization import background_intensity


def main(args):
    # Allow default parameters to work.  Assume None is only
    # meaningful as a default value
    other_args = set(['returnParameterFile', 'scheduler_address'])
    kwargs = {k: v for k, v in vars(args).items()
              if k not in other_args and v is not None}
    Client(args.scheduler_address)
    I_0 = background_intensity(**kwargs)
    with open(args.returnParameterFile, 'w') as f:
        f.write('BackgroundIntensity = ' + ','.join(map(str, I_0)) + '\n')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
