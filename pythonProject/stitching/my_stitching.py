import os
import time
import typing
from dataclasses import dataclass

import cv2
from enum import Enum
import imutils
import numpy as np

from pythonProject.camera.camera_calibration import CamIntrinsics
from pythonProject.helpers.image_manipulation import load_images


class KeypointFeatures(Enum):
    """Enum representing keypoint detection and matching method."""
    ORB  = 0
    SIFT = 1


@dataclass
class CVImageWithKeypoints:
    """Dataclass holding image with corresponding keypoints and descriptors."""
    img: np.ndarray
    keypoints: typing.List[cv2.KeyPoint]
    descriptors: np.ndarray


class MyStitcher:

    def __init__(self, keypoint_features: KeypointFeatures = KeypointFeatures.SIFT, debug = False):

        self.debug = debug

        if KeypointFeatures.SIFT == keypoint_features:
            # Initiate SIFT detector
            self.kp_detector = cv2.SIFT_create()
            self.matching = self.sift_matching
        else:
            # Create ORB detector with 5000 features.
            self.kp_detector = cv2.ORB_create(5000)
            self.matching = self.orb_matching

    def find_keypoints_and_descriptors(self, image: typing.Union[CVImageWithKeypoints, np.ndarray], ref_mask=None) \
            -> (np.ndarray, list[cv2.KeyPoint], np.ndarray):
        """Unrolls keypoints and descriptors from CVImageWithMatches or extracts them from image using Keypoint detector.
        :param image: Input image with or without keypoints and descriptors
        :param kp_detector: Keypoint detector
        :param ref_mask: bitmap mask for selection of region, where to find reference image features
        :return: cv2.Mat image, keypoints, descriptors
        """

        # Use buffered keypoints and descriptors for image if available
        if isinstance(image, CVImageWithKeypoints):
            kp, d = image.keypoints, image.descriptors
            image = image.img
        else:  # Otherwise extract them from image using kp_detector
            # Check if colored image was supplied
            if len(image.shape) == 3:
                # Convert to grayscale.
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image
            kp, d = self.kp_detector.detectAndCompute(image_gray, ref_mask)

        return image, kp, d

    def stitch(self, imgs: list[typing.Union[np.ndarray, CVImageWithKeypoints]], ref_mask: np.ndarray = None,
               th_imgs: list[np.ndarray] = []) -> [np.ndarray, list[np.ndarray]]:
        """Finds homographies between supplied images and stitches them.
        Recursively stitches one image after another to the reference.

        :param imgs: List of images to stitch. The first image is the reference.
        Data types in list can be either cv2.Mat images or CVImageWithMatches dataclasses.
        :param ref_mask: bitmap mask for selection of region, where to find reference image features
        :param th_imgs: Homography transformations of individual stitched images in the panorama.
        Argument is present for recursion calls. It is not expected to be supplied from outside.
        :param matching: Keypoint detection and matching algorithm to use
        :return: cv2.Mat Panorama, Homographies of individual images to final panorama
        """

        ref_img = imgs.pop(0)
        img = imgs.pop(0)

        ref_img, kp1, d1 = self.find_keypoints_and_descriptors(ref_img, ref_mask)
        img, kp2, d2 = self.find_keypoints_and_descriptors(img)

        if self.debug:
            cv2.imshow("Preprocessed REF_IMG", imutils.resize(ref_img, width=1928))
            cv2.waitKey()
            cv2.destroyWindow("Preprocessed REF_IMG")
            cv2.imshow("Preprocessed IMG", imutils.resize(img, width=1928))
            cv2.waitKey()
            cv2.destroyWindow("Preprocessed IMG")

        matches = self.matching(d1, d2)

        homography, mask = find_homography(kp1, kp2, matches)

        if self.debug:
            matchesMask = mask.ravel().tolist()
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            img_keypoints_matches = cv2.drawMatches(ref_img, kp1, img, kp2, matches, None, **draw_params)
            cv2.imshow("Keypoints and Matches", imutils.resize(img_keypoints_matches, width=1928))
            cv2.waitKey()
            cv2.destroyWindow("Keypoints and Matches")

        out_image, th, th_img = concatenate_images(ref_img, img, homography)
        # cv2.imshow("Stitched Img", imutils.resize(outImage, width=1928))
        # cv2.waitKey()

        # Buffer keypoints and descriptors for ref_img and img
        # ToDo: Use only those keypoints from img, which do not have correspondence in ref_img
        buffered_keypoints = transform_keypoints(kp1, th) + transform_keypoints(kp2, th_img)
        if self.debug:
            print(f"No of buffered keypoints: {len(buffered_keypoints)}")
        buffered_descriptors = np.vstack((d1, d2))

        # Add transform from the last stitching to the previous image transformations.
        if len(th_imgs) > 1:
            for idx in range(len(th_imgs)):
                th_imgs[idx] = th.dot(th_imgs[idx])
        else:
            th_imgs.append(th)
        th_imgs.append(th_img)

        if len(imgs) == 0:  # Exit point from recursion. When no image is left for processing
            return CVImageWithKeypoints(out_image, buffered_keypoints, buffered_descriptors), th_imgs,
        else:  # Recursively apply this function to all images in input list
            if ref_mask is not None:
                pano_h, pano_w = out_image.shape[:2]
                ref_mask = cv2.warpPerspective(ref_mask, np.float32(th), (pano_w, pano_h))
            return self.stitch([CVImageWithKeypoints(out_image, buffered_keypoints, buffered_descriptors), *imgs],
                               ref_mask, th_imgs)

    @classmethod
    def orb_matching(cls, d1: np.ndarray, d2: np.ndarray) -> list[cv2.DMatch]:
        """Finds matches between 2 sets of descriptors, resp keypoints.
        Uses cv2.BFMatcher for keypoints detected by ORB algorithm.

        :param d1: keypoints descriptions for image1
        :param d2: keypoints descriptions for image2
        :return: list[matches]
        """

        # Match features between the two images.
        # Brute Force matcher with Hamming distance as measurement mode.
        # Do not use crossCheck=True with Lowe's ratio test.
        # Cross-checking enforces mutual agreement for a single match,
        # which conflicts with the concept of considering multiple neighbors.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        # matches = matcher.knnMatch(d1, d2, k=2)
        #
        # # Apply Lowe's Ratio Test
        # good_matches = []
        # for m, n in matches:
        #     if m.distance < 0.75 * n.distance:  # 0.75 is the ratio threshold
        #         good_matches.append(m)
        #
        # return good_matches

        # Match the two sets of descriptors.
        matches = matcher.match(d1, d2)
        # Sort matches on the basis of their Hamming distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches) * 0.9)]

        return matches

    @classmethod
    def sift_matching(cls, d1: np.ndarray, d2: np.ndarray) -> list[cv2.DMatch]:
        """Finds matches between 2 sets of descriptors, resp keypoints.
        Uses cv2.FlannBasedMatcher for keypoints detected by SIFT algorithm.

        :param d1: keypoints descriptions for image1
        :param d2: keypoints descriptions for image2
        :return: list[matches]
        """

        # FLANN parameters for floating-point descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # Number of checks for tree traversal

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(d1, d2, k=2)

        # Apply Lowe's Ratio Test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # 0.75 is the ratio threshold
                good_matches.append(m)

        return good_matches


def find_homography(ref_img_keypoints: list[cv2.KeyPoint], img_keypoints: list[cv2.KeyPoint],
                    matches: list[cv2.DMatch]) -> (np.ndarray, np.ndarray):
    """Find homography of img to ref_img.

    :param ref_img_keypoints: Keypoints of Reference image
    :param img_keypoints: Keypoints of Image to be transformed
    :param matches: list of Matches between Keypoints
    :return: Homography transformation, matches mask
    """

    no_of_matches = len(matches)
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(no_of_matches):
        p1[i, :] = ref_img_keypoints[matches[i].queryIdx].pt
        p2[i, :] = img_keypoints[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p2, p1, cv2.RANSAC)

    return homography, mask


def transform_keypoints(keypoints, homography):
    """Transform all keypoints using homography."""

    pts = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(pts, homography)
    transformed_keypoints = [cv2.KeyPoint(pt[0][0], pt[0][1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                             for pt, kp in zip(transformed_pts, keypoints)]
    return transformed_keypoints


def concatenate_images(ref_img, img, homography):
    """Concatenate or add two images.
    The function computes minimum image shape necessary to avoid ref_img or img cropping.

    :param ref_img: Reference image
    :param img: Image to be warped to ref_img plane
    :param homography: Transformation for img warping
    """

    h, w = img.shape[:2]
    corners_img = [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]  # .reshape(-1, 1, 2)
    corners_img_dst = cv2.perspectiveTransform(np.float32([corners_img]), homography)
    # Find the bounding rectangle
    bx, by, bwidth, bheight = cv2.boundingRect(corners_img_dst)
    # Compute the translation homography that will move (bx, by) to (0, 0)
    th = np.array([
        [1, 0, -bx],
        [0, 1, -by],
        [0, 0, 1]
    ])

    # Find bounding rect for stitched image
    h, w = ref_img.shape[:2]

    corners_ref = [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]  # .reshape(-1, 1, 2)
    corners_dst = cv2.perspectiveTransform(np.float32([corners_ref]), th)
    # Find the bounding rectangle
    rbx, rby, rbwidth, rbheight = cv2.boundingRect(corners_dst)
    thr = np.array([
        [1, 0, -rbx],
        [0, 1, -rby],
        [0, 0, 1]
    ])

    # Compute the translation homography that will move (rbx, rby) to (0, 0)
    pano_x = int(min(0, bx))
    pano_y = int(min(0, by))
    th_pano = np.array([
        [1, 0, -pano_x],
        [0, 1, -pano_y],
        [0, 0, 1]
    ])

    # Combine the homographies
    th = thr.dot(th_pano.dot(th))
    th_img = th.dot(homography)

    corners_dst = cv2.perspectiveTransform(np.float32([corners_img]), th_img)
    # Find the bounding rectangle
    bx, by, bwidth, bheight = cv2.boundingRect(corners_dst)
    corners_dst = cv2.perspectiveTransform(np.float32([corners_ref]), th)
    # Find the bounding rectangle
    rbx, rby, rbwidth, rbheight = cv2.boundingRect(corners_dst)

    pano_w = int(max(bx + bwidth, rbx + rbwidth))
    pano_h = int(max(by + bheight, rby + rbheight))

    # Apply transformations to images
    foreground = cv2.warpPerspective(img, th_img, (pano_w, pano_h))
    background = cv2.warpPerspective(ref_img, np.float32(th), (pano_w, pano_h))

    # Create masks for img addition
    _, foreground_mask = cv2.threshold(cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)
    _, background_mask = cv2.threshold(cv2.cvtColor(background, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)

    overlay_mask = cv2.bitwise_and(foreground_mask, background_mask)
    overlay_mask_3channel = cv2.merge([overlay_mask, overlay_mask, overlay_mask])
    no_overlay_mask = cv2.bitwise_not(overlay_mask)

    # cv2.imshow("Mask", imutils.resize(overlay_mask, width=1928))
    # cv2.waitKey()
    # Add the masked foreground and background.

    outImage = cv2.add(foreground, background, mask=no_overlay_mask)
    # Do not augment reference image
    # outImage = cv2.add(outImage, background, mask=overlay_mask, dst=outImage)
    # Blend reference image in overlap
    outImage = cv2.add(outImage, cv2.addWeighted(cv2.bitwise_and(foreground, overlay_mask_3channel), 0.5,
                                                 cv2.bitwise_and(background, overlay_mask_3channel), 0.5,
                                                 0))

    return outImage, th, th_img


def preprocess_images(imgs: list[np.ndarray]):
    intrinsics = CamIntrinsics(os.path.join("./camera", "Ezviz_C6N"))

    imgs = intrinsics.undistort(imgs)
    # imgs = intrinsics.spherical_warp(imgs)

    return imgs


# Convert a list of cv2.KeyPoint objects to a serializable format
def keypoints_to_list(keypoints):
    return [ (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints ]

# Convert a list back to cv2.KeyPoint objects
def list_to_keypoints(keypoint_list):
    return [ cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
             for (pt, size, angle, response, octave, class_id) in keypoint_list
           ]


if __name__ == "__main__":
    DEBUG = False
    my_stitcher = MyStitcher(keypoint_features=KeypointFeatures.SIFT, debug=DEBUG)

    imgs, _ = load_images(["../imgs/testing/night_test_1"])
    imgs = preprocess_images(imgs)

    imgs[0], imgs[1] = imgs[1], imgs[0]

    # ref_mask = create_mask(imgs[0])
    ref_mask = None #cv2.imread("base_rect_mask.jpg", cv2.IMREAD_GRAYSCALE)
    t_start = time.perf_counter()

    if DEBUG:
        for img in imgs:
            cv2.imshow("Preprocessed IMG", imutils.resize(img, width=1928))
            cv2.waitKey()
        cv2.destroyWindow("Preprocessed IMG")

    pano_with_keypoints, transforms = my_stitcher.stitch(imgs)
    pano = pano_with_keypoints.img
    print(f"Stitching Time: {time.perf_counter() - t_start}")

     #Test that image transform were computed correctly
    if DEBUG:
        h, w = imgs[0].shape[:2]
        corners_ref = [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]  # .reshape(-1, 1, 2)
        for t in transforms:
            corners_img = cv2.perspectiveTransform(np.float32([corners_ref]), t)
            pano = cv2.polylines(pano, [np.int32(corners_img)], True, (255, 255, 255), 3, cv2.LINE_AA)

    # Display image
    cv2.imshow("Pano Plane", imutils.resize(pano, width=1280))
    cv2.imwrite("pano.jpg", pano)

    # intrinsics = CamIntrinsics(os.path.join("./camera", "Ezviz_C6N"))
    # pano = intrinsics.cylindrical_warp(pano)
    # # Display image
    # cv2.imshow("Pano Warp", imutils.resize(pano, width=1280))
    # cv2.imwrite("pano_warp.jpg", pano)

    cv2.waitKey(0)
    cv2.destroyAllWindows()