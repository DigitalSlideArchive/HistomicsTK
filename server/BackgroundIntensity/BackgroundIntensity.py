from ctk_cli import CLIArgumentParser

from histomicstk.preprocessing.color_normalization import background_intensity


def main(args):
    # Allow default parameters to work.  Assume None is only
    # meaningful as a default value
    kwargs = {k: v for k, v in vars(args).items()
              if k != 'returnParameterFile' and v is not None}
    I_0 = background_intensity(**kwargs)
    with open(args.returnParameterFile, 'w') as f:
        f.write('BackgroundIntensity = ' + ','.join(map(str, I_0)) + '\n')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
