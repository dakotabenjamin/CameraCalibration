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
import argparse
from subprocess import call

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
             description='Remove distortion from photos with previously '
                         'generated parameters')
    parser.add_argument('images', help='path to images')
    parser.add_argument('-m', '--matrix', required=True, 
                        help='path to the matrix.txt file')
    parser.add_argument('-d', '--distortion', required=True,
                        help='path to distortion matrix')
    args = parser.parse_args()

    # get image path
    if os.path.isdir(args.images):
        images_path = args.images
    else:
        logging.error('images path is not valid')
        exit()
    # get images in list
    extensions = ['jpg', 'JPG', 'png']

    images = [fn for fn in os.listdir(images_path)
              if any(fn.endswith(ext) for ext in extensions)]

    out_path = os.path.join(images_path, 'out')

    # load parameters to np arrays
    if os.path.isfile(args.matrix):
        mpath = os.path.abspath(args.matrix)
        K = np.loadtxt(mpath)
    else:
        logging.error('matrix path is not valid')
        exit()
    if os.path.isfile(args.distortion):
        dpath = os.path.abspath(args.distortion)
        d = np.loadtxt(dpath)
    # d[2:] = 0  # We only need first two values

    logging.info("Matrix: \n" + str(K))
    logging.info("Distortion: " + str(d))

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    logging.debug("Starting loop")
    logging.debug("writing images to {0})".format(out_path))
    for image in images:
        # logging.debug("var %s", image)
        # imgname = os.path.split(image)[1] 
        logging.debug("Undistorting %s . . . ", image)
        # read one of your images
        path = os.path.join(images_path, image)
        img = cv2.imread(path)
        h, w = img.shape[:2]

        # un-distort
        newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)
        newimg = cv2.undistort(img, K, d, None, newcamera)

        # TODO Write exif: camera maker, camera model, focal length?, GPS to new file
        # cv2.imwrite("original.jpg", img)
        newimg_path = os.path.join(out_path, image)
        cv2.imwrite(newimg_path, newimg)

        # Only works on linux
        command = "exiftool -TagsFromFile {0} -all:all {1} -overwrite_original ".format(os.path.abspath(path), os.path.abspath(newimg_path)).split(' ')
        try:
            logging.debug(command)
            call(command)
        except Exception as err:
            logging.error("Failed to write to exif: {0}".format(err))
