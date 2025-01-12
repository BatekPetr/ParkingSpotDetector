'''
Script for panorama creation and parking slot detection. Uses camera to take pictures and process them.
'''

import argparse

from stitching import stitch_and_detect


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cam_stitch_and_detect.py',
                                     description='Stitch images taken by camera '
                                                 'and perform available parking slot detection.')
    parser.add_argument('--save_name', nargs='?', type=str, default=None, required=False,
                        help="Specify name under which images taken by camera are going to be saved. "
                             "If None or not specified, images are not saved.")

    __doc__ += '\n' + parser.format_help()
    print(__doc__)

    args = parser.parse_args()
    args.img = None

    stitch_and_detect.main(args)