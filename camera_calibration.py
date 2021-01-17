import numpy as np
import cv2
import pickle
import glob
import os
import sys


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
    objpoints = [] # 3d points in real world space (once there're defined always the same points)
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

def undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        if (sys.argv[1] == "calibrate"):
            images = glob.glob('camera_cal/calibration*.jpg')
            mtx, dist = calibrate_camera(9,6,images,)
            dist_pickle = {}
            dist_pickle["mtx"] = mtx
            dist_pickle["dist"] = dist
            pickle.dump( dist_pickle, open(os.sep.join([os.path.dirname(os.path.realpath(__file__)), "calibration.p"]), "wb"))
        elif (sys.argv[1] == "undist"):
            img = cv2.imread('camera_cal/calibration1.jpg')
            dist_pickle = pickle.load(open(os.sep.join([os.path.dirname(os.path.realpath(__file__)), "calibration.p"]), "rb" ) )
            mtx = dist_pickle["mtx"]
            dist = dist_pickle["dist"]
            dst = undistort(img, mtx, dist)
            cv2.imshow("Undistorted", dst)
            cv2.imwrite('output_test_images/calibration1_undistort.jpg',dst)            
    else:
        print("camera_calibration.py need arguments")
    
    