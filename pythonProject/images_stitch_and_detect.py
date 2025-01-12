'''
Script for panorama creation and parking slot detection. Uses images loaded from disk.
'''

import argparse

from pythonProject.stitching import stitch_and_detect


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='images_stitch_and_detect.py',
                                     description='Stitch images loaded from files and perform available parking'
                                                 ' slot detection.')
    parser.add_argument('--img', nargs='+', help='input images')

    __doc__ += '\n' + parser.format_help()
    print(__doc__)

    args = parser.parse_args()
    args.save_name = None

    if args.img is None:
        # Please choose one of the options below, or specify it as a script argument
        # --------------------------------------------------------------------------
        # args.img = ["../imgs/testing/day_demo_1"]
        # args.img = ["../imgs/testing/day_demo_2_0.jpg",
        #             "../imgs/testing/day_demo_2_1.jpg",
        #             "../imgs/testing/day_demo_2_2.jpg"]
        # args.img = ["../imgs/testing/day_demo_3"]

        # args.img = ["../imgs/testing/sunny_day_1"]
        args.img = ["../imgs/testing/sunny_day_2"]
        # args.img = ["../imgs/testing/sunny_day_3"]

        # args.img = ["../imgs/testing/night_demo_1"]
        # args.img = ["../imgs/testing/night_demo_2"]
        # args.img = ["../imgs/testing/night_demo_3"]

        # args.img = ["../imgs/testing/pano_4_imgs"]

    stitch_and_detect.main(args)
