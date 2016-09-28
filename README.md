# CameraCalibration

Usage Instructions:
```
# Calibrate Checkerboard:
python opencv_calibrate.py /path/*.jpg

# This will produce two files. Use them in the following script:
python undistort.py --matrix matrix.txt --distortion distortion.txt /path/*.jpg
```

# Docker Usage

This assumes you already have the distortion and matrix parameters. 

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
