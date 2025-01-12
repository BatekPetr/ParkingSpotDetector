import os.path
import typing

import numpy as np
import cv2
import glob


class CamIntrinsics:

    def __init__(self, params_directory=None):

        if params_directory:
            self.load_params(params_directory)
        else:
            print("No parameters supplied. Please calibrate the camera.")
            self.mtx = None
            self.dist = None

    def calibrate(self, images_path_names, n_rows, n_cols,
                  params_save_directory=None, draw_corners=False) -> bool:
        """
        Calculates camera intrinsics and lens distortions from the set of chessboard images.

        :param images_path_names: Chessboard image files including path
        :param n_rows: Chessboard rows pattern
        :param n_cols: Chessboard cols pattern
        :param params_save_directory: Directory, where computed params are to be saved.
                                            If None, parameters won't be saved
        :param draw_corners: Boolean whether to draw detected corners in images
        :return: bool Success status
        """

        # init return variable
        success = True

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((n_rows * n_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for fname in images_path_names:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (n_cols, n_rows), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                if draw_corners:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (n_cols, n_rows), corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

        cv2.destroyAllWindows()

        # Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            self.mtx = mtx
            self.dist = dist
            if params_save_directory is not None:
                np.savetxt(os.path.join(params_save_directory, "intrinsics.txt"), mtx)
                np.savetxt(os.path.join(params_save_directory, "distortions.txt"), dist)

        else:
            success = False

        return success

    def load_params(self, directory) -> bool:
        """
        Load intrinsics and distortions parameters from files.

        :param directory: directory, where params files are located
        :return: bool success status
        """
        try:
            self.mtx = np.loadtxt(os.path.join(directory, "intrinsics.txt"))
            self.dist = np.loadtxt(os.path.join(directory, "distortions.txt"))
            return True
        except Exception as e:
            print(e)
            print("Parameters are not present in the expected format.")
            return False

    def undistort_img(self, img):
        """
        Undistorts/rectifies image.

        :param img: cv2.Mat image
        :return: undistorted cv2.Mat image
        """
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

        # undistort
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst

    def undistort(self, imgs #: typing.Union[cv2.typing.MatLike | list[cv2.typing.MatLike]]
                  ):
        """
        Undistorts/rectifies images in a list.

        :param imgs: list of cv2.Mat image
        :return: list of undistorted cv2.Mat images
        """

        if isinstance(imgs, list):
            out_imgs = []
            for img in imgs:
                out_imgs.append(self.undistort_img(img))
            return out_imgs
        else:
            return self.undistort_img(imgs)

    def cylindrical_warp_img(self, img):
        focal_length = (self.mtx[0,0] + self.mtx[1,1])/2.0
        h, w = img.shape[:2]
        # Define the center of the image
        cx, cy = w // 2, h // 2

        # Create a map for the cylindrical projection
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # Convert to cylindrical coordinates
                theta = (x - cx) / focal_length
                h_ = (y - cy) / focal_length
                x_ = focal_length * np.tan(theta) + cx
                y_ = focal_length * h_ / np.cos(theta) + cy

                # Map coordinates
                map_x[y, x] = x_
                map_y[y, x] = y_

        # Remap the image
        warped_image = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return warped_image

    def cylindrical_warp(self, imgs):
        if isinstance(imgs, list):
            out_imgs = []
            for img in imgs:
                out_imgs.append(self.cylindrical_warp_img(img))
            return out_imgs
        else:
            return self.cylindrical_warp_img(imgs)

    def spherical_warp_img(self, image):
        focal_length = (self.mtx[0,0] + self.mtx[1,1])/2.0
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2  # Principal point

        # Create mapping arrays
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # Convert to spherical coordinates
                theta = (x - cx) / focal_length
                phi = (y - cy) / focal_length

                # Map to Cartesian coordinates on the sphere
                x_sphere = focal_length * np.tan(theta) + cx
                y_sphere = focal_length * (np.sin(phi) / np.cos(theta)) + cy

                # Set the mapping
                map_x[y, x] = x_sphere
                map_y[y, x] = y_sphere

        # Perform the remapping
        warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return warped_image

    def spherical_warp(self, imgs):
        if isinstance(imgs, list):
            out_imgs = []
            for img in imgs:
                out_imgs.append(self.spherical_warp_img(img))
            return out_imgs
        else:
            return self.spherical_warp_img(imgs)


if __name__=="__main__":
    images = glob.glob('../imgs/calibration/*.jpg')

    # Chessboard pattern
    N_ROWS = 6
    N_COLS = 8

    camera_calibration = CamIntrinsics()
    ret = camera_calibration.calibrate(images, N_ROWS, N_COLS, "../Ezviz_C6N")
    if ret:
        print("Camera calibration succesful.")
    else:
        print("A problem appeared during calibration.")