# CameraCalibration

This suite of scripts will find camera matrix and distortion parameters with a set of checkerboard images, then use thos parameters to remove distortion from photos. 

## Installation

You need to install numpy and opencv:
```
pip install numpy
sudo apt-get install python-opencv
```

## Usage: Calibrate chessboard

First you will need to take some photos of a black and white chessboard, [like this one](http://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf).

Then you will run the `opencv_calibrate.py` script to generate the matrix and distortion files. 
```
python opencv_calibrate.py ./sample/chessboard/ 9 6
```
The first argument is the path to the chessboard. You will also have to input the chessboard dimensions (the number of squares in x and y) Optional arguments:
```
--out           path      if you want to output the parameters and the image outputs to a specific path. otherwise it gets writting to ./out
--square_size   float     if your chessboard squares are not square, you can change this. default is 1.0
```
## Usage: undistort photos
With the photos and the produced matrix.txt and distortion.txt, run the following:

```
python undistort.py --matrix matrix.txt --distortion distortion.txt "/path/*.jpg"
```

**Note**: Do not forget the quotes in "/path/*.jpg"

### Docker Usage for undistorting images

This assumes you already have the distortion and matrix parameters. Put the matri.txt and distortion.txt in their own directory (eg. sample/config) and do the following:

Build: 
```
docker build -t cc_undistort .
```

Run: (using sample images in this example)
```
docker run -v ~/CameraCalibration/sample/images:/app/images \
           -v ~/CameraCalibration/sample/config:/app/config \
           cc_undistort
```
