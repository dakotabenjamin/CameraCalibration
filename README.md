# CameraCalibration

Usage Instructions:
```
# Calibrate Checkerboard:
python calibrate.py /path/*.jpg

# This will produce two files. Use them in the following script:
python undistort.py --matrix matrix.txt --distortion distortion.txt /path/*.jpg
