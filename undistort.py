#!/usr/bin/env python

"""
image undistortion for cameras with calibration information.
reads distorted images and their calibration matrix and distortion coefficients
and writes undistorted photos
usage:
    undistort.py --matrix <matrix path> --distortion <distortion path> <image path>

Code forked from Jan Erik Solem's blog:
http://www.janeriksolem.net/2014/05/how-to-calibrate-camera-with-opencv-and.html
"""

import numpy as np
import cv2
import glob
import os
import logging
import sys
import getopt
from subprocess import call

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


if __name__ == '__main__':
    args, img_path = getopt.getopt(sys.argv[1:], '', ['matrix=', 'distortion='])
    args = dict(args)
    args.setdefault('--matrix', os.path.abspath('./sample/output/matrix.txt'))
    args.setdefault('--distortion', os.path.abspath('./sample/output/distortion.txt'))
    if not img_path:
    #    img_path = os.path.abspath('./sample/img/*.JPG')  # default
        img_path = glob.glob(os.path.abspath('./sample/img/*.JPG'))  # default
    #else:
    #    img_path = img_path[0]
    images_path = os.path.split(img_path[0])[0]

    matrix_path = args.get('--matrix')
    distortion_path = args.get('--distortion')
    out_path = os.path.join(images_path, 'out')

    # load parameters to np arrays
    K = np.loadtxt(matrix_path)
    d = np.loadtxt(distortion_path)
    # d[2:] = 0  # We only need first two values

    print("Matrix: \n" + str(K))
    print("Distortion: " + str(d))

    # copy parameters to arrays
    # K = np.array([[1743.23312, 0, 2071.06177], [0, 1741.57626, 1476.48298], [0, 0, 1]])
    # d = np.array([-0.307412748, 0.300929683, 0, 0, 0])  # just use first two terms (no translation)

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    logging.debug("Starting loop")
    logging.debug("writing images to {0})".format(out_path))
    for image in img_path:
        # logging.debug("var %s", image)
        imgname = os.path.split(image)[1] 
        logging.debug("Undistorting %s . . . ", imgname)
        # read one of your images
        img = cv2.imread(image)
        h, w = img.shape[:2]

        # un-distort
        newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)
        newimg = cv2.undistort(img, K, d, None, newcamera)

        # TODO Write exif: camera maker, camera model, focal length?, GPS to new file
        # cv2.imwrite("original.jpg", img)
        newimg_path = os.path.join(out_path, imgname)
        cv2.imwrite(newimg_path, newimg)

        # Only works on linux
        command = "exiftool -TagsFromFile {0} -all:all {1}".format(image, newimg_path).split(' ')
        call(command)
