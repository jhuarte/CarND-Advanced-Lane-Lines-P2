# **Project 2: CarND-Advanced-Lane-Lines** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# Overview
In this project, the goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Camera calibration

The objective of this step is to compute the camera calibration matrix `(mtx)` and  the distorsion coefficients `(dist)`. Four main functions are used:

1. `calibrate_camera(nx,ny,images)`: Calibrates the camera
2. `undistort(img, mtx, dist)`: Undistort the image
3. `read_calibration()`: Read the calibration parameters calculated and stored in the first step

# 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image

The fucntion `calibrate_camera(nx,ny,images)` calibrates a camera using the nx,ny parameters (chessboard pattern) and a directory with images of this pattern, where `nx` is the number of corners in a row, `ny` is the number of corners in a column and `image` is the list of the images used to calculate the calibration. The function `cv2.findChessboardCorners` is used to detect the corners, the function `drawChessboardCorners` to draw the corners detected and the `calibrateCamera` to calibrate. Only images where the chessboard is detected is used to compute the calibration.

<img src="output_test_images/calibration_chessboard.png"/>

The code used to calibrate the camera:
```python
def calibrate_camera(nx,ny,images):
    """
    Calibrate a camera using the nx,ny parameters (chessboard pattern) and a directory with images of this pattern
    nx: number of corners in a row
    ny: number of corners in a column
    image: list of the images used to calculate the calibration
    """
    # Prepare object points [(0,0,0), (1,0,0), ....,(6,5,0)] using the nx and ny parameters (chessboard)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space (once there're defined, they are always the same points)
    imgpoints = [] # 2d points in image plane (for each image used to calibrate).

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to gray
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    if (len(objpoints) > 0):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist
```

An example image (one chessboard) is undistorted using the function `undistort(img, mtx, dist)` that undistort the image `img` using the calibrarion matrix `mtx` and the distorsion parameters `dist`. Return the undirstoted image `dst`.

<img src="camera_cal/calibration1.jpg" width="240"/> <img src="output_test_images/calibration1_undistort.jpg" width="240"/>

# Pipeline (test images)

## 1. Provide an example of a distortion-corrected image.

In this step I apply distortion correction to the images placed on the folder `test_images` using the next code.

```python
# ----------------------------------------------------------------------------------------------------
# Undistort all the test_images folder
# ----------------------------------------------------------------------------------------------------

# Read calibration matrix (mtx) and distortion coefficients
mtx, dist = read_calibration("calibration.p")
# Read test_images folder
images = [cv2.imread(fname) for fname in glob.glob("test_images/*")]

# All the images from the test_image folder are undirstorted but only one displayed as an example
undistorted_images = list(map(lambda img: undistort_image_visualization(img,mtx,dist), images))
```
The result are the following undirstorted images.

<img src="output_test_images/calibration1_undistort.jpg" width="240"/>


## 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

On this step I created a pipeline to obtain a thresholded binary image combining the result from color transforms (`RGB`, `HLS` and `LAB` color spaces) and gradients (directional gradient, magnitude and direction). 

The functions that create a binary image using the gradient are listed below with the result applied to the same image. Latter we will see how all this functions are combined in one in order to speed up the final pipeline:

* `abs_sobel_thresh(gray, orient = 'x', grad_kernel = 3, grad_thresh = (0, 255)`: Creates a binary image using applying the sobel absolute operator in one axes (`x` or `y`)

<img src="output_test_images/sobel_operator.png">

* `mag_threshold(gray, mag_kernel = 3, mag_thresh = (0, 255))`: Creates a binary image using applying the gradient magnitude

<img src="output_test_images/gradient_magnitude.png">

* `dir_threshold(gray, dir_kernel = 3, dir_thresh = (0, np.pi/2))`: Creates a binary image using applying the gradient direction

<img src="output_test_images/gradient_direction.png">


Also a color threshold is used to detect the lines. As we have lines with different colors, we have shadows on the road and other challenges situations, I've applied the thresholds over different color spaces to make the final pipeline more robust.

* `R` channel of the `RGB` color space 
* `S` channel of the `HLS` color space
* `L` channel of the `HLS` color space
* `B` channel of the `LAB` color space

The next figure display each channel of the color spaces used.

<img src="output_test_images/gradient_direction.png">

To apply the threshold of each channel I created this functions:
* `color_hls_threshold(img, h_thresh=(255,0),l_thresh=(255,0),s_thresh=(255,0))`: Applies a color threshold on the HLS channels. With the default threshold parameters channels are "disabled"

<img src="output_test_images/hls_threshold.png">

* `color_rgb_threshold(img, r_thresh=(255,0),g_thresh=(255,0),b_thresh=(255,0))`: Applies a color threshold on the RGB channels. With the default threshold parameters channels are "disabled"

<img src="output_test_images/rgb_threshold.png">

* `color_lab_threshold(img, l_thresh=(255,0), a_thresh=(255,0), b_thresh=(255,0))`: Applies a color threshold on the LAB channels. With the default threshold parameters channels are "disabled"

<img src="output_test_images/lab_threshold.png">


The `combined_threshold(gray,kernel,gradx_thr,grady_thr,mag_thr,dir_thr)` function wraps all of the above functions/steps and return a binary thresholded image `combined_binary` and  `color_binary` (an stacked combination of binary images only for test). The function which is the function called on the final **pipeline**.    

## Pipeline (video)


## Discussion

# Licence
[MIT](https://choosealicense.com/licenses/mit/)

# Contribute
Pull requests are welcome. For major changes, please open an issue first to discuss.

# About me
My name is [Javier Huarte](https://github.com/jhuarte) @jhuarte. I'm a Computer Science Engineer by EUPLA & UOC Univerties. Robotics, coding, cycling and motorsport apossionated. Actually R&D Engineer at [ITAINNOVA](www.itainnova.es).
