from ctk_cli import CLIArgumentParser
from dask.distributed import Client

from histomicstk.preprocessing.color_normalization import background_intensity


def main(args):
    other_args = set(['returnParameterFile', 'scheduler_address'])
    kwargs = {k: v for k, v in vars(args).items()
              if k not in other_args}
    # Allow (some) default parameters to work.  Assume certain values
    # are not valid.
    for k in 'sample_percent', 'sample_approximate_total':
        if kwargs[k] == -1:
            del kwargs[k]

    Client(args.scheduler_address or None)
    I_0 = background_intensity(**kwargs)
    with open(args.returnParameterFile, 'w') as f:
        f.write('BackgroundIntensity = ' + ','.join(map(str, I_0)) + '\n')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
