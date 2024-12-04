import os

import cv2
from enum import Enum
import imutils
import numpy as np

from pythonProject.camera.camera_calibration import CamIntrinsics
from pythonProject.camera.undistort_imgs import undistort_images_from_files


class Matching(Enum):
    ORB  = 0
    SIFT = 1


def orb_matching(ref_img: np.ndarray, img: np.ndarray, ref_mask: np.ndarray = None):
    """Finds keypoints and matches using ORB detector.

    :param ref_img: Reference image
    :param img: image to be transformed
    :param ref_mask: bit mask of reference image region for feature finding
    :return: ref_img keypoints, img keypoints, matches
    """
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(ref_img, ref_mask)
    kp2, d2 = orb_detector.detectAndCompute(img, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]

    return kp1, kp2, matches


def sift_matching(ref_img: np.ndarray, img: np.ndarray, ref_mask: np.ndarray = None):
    """Finds keypoints and matches using SIFT detector.

    :param ref_img: Reference image
    :param img: image to be transformed
    :param ref_mask: bit mask of reference image region for feature finding
    :return: ref_img keypoints, img keypoints, matches
    """
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, d1 = sift.detectAndCompute(ref_img, ref_mask)
    kp2, d2 = sift.detectAndCompute(img, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return kp1, kp2, good


def find_homography_to_base(ref_img: np.ndarray, img: np.ndarray, ref_mask: np.ndarray = None,
                            matching: Matching = Matching.SIFT, draw_mathes: bool = False):
    """Find homography of img to ref_img.

    :param ref_img: Reference image
    :param img: image to be transformed
    :param ref_mask: bit mask of reference image region for feature finding
    :param matching: matching algorithm to use
    :param draw_mathes: indicates, whether to draw mathes
    :return: Homography transformation, matches mask
    """

    # Check if colored image was supplied
    if len(ref_img.shape) == 3:
        # Convert to grayscale.
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    if len(img.shape) == 3:
        # Convert to grayscale.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find features and matches
    if Matching.SIFT == matching:
        kp1, kp2, matches = sift_matching(ref_img, img, ref_mask)
    else:
        kp1, kp2, matches = orb_matching(ref_img, img, ref_mask)

    no_of_matches = len(matches)
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p2, p1, cv2.RANSAC)

    if draw_mathes:
        matchesMask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img_keypoints_matches = cv2.drawMatches(ref_img, kp1, img, kp2, matches, None, **draw_params)
        cv2.imshow("Keypoints and Matches", imutils.resize(img_keypoints_matches, width=1928))

    return homography, mask

def my_stitch(imgs: list[np.ndarray], ref_mask: np.ndarray = None, out_th_imgs: list[np.ndarray] = [])\
        -> [np.ndarray, list[np.ndarray]]:
    """Finds homographies between supplied images and stitches them.
    Recursively stitches one image after another to the reference

    :param imgs: List of images to stitch. The first image is the reference
    :param ref_mask: bitmap mask for selection of region, where to find reference image features
    :param out_th_imgs: Homography transformations of individual stitched images in the final panorama.
    Argument is present for recursion calls. It is not expected to be supplied from outside.
    :return: cv2.Mat Panorama, Homoghraphies of individual images to final panorama
    """
    ref_img = imgs.pop(0)
    img = imgs.pop(0)
    homography, mask = find_homography_to_base(ref_img, img, ref_mask)

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

    # Compute the translation homography that will move (rx, ry) to (0, 0)
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

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img, th_img, (pano_w, pano_h))
    # transformed_img = cv2.polylines(transformed_img, [np.int32(cv2.perspectiveTransform(corners_img_dst, th))],
    #                                 True, (255, 255, 255), 3, cv2.LINE_AA)

    foreground = transformed_img # cv2.warpPerspective(transformed_img, np.float32(th_img), (pano_w, pano_h))
    _, foreground_mask = cv2.threshold(cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)
    #foreground_mask = cv2.fillPoly(foreground_mask, [np.int32(cv2.perspectiveTransform(corners_img_dst, th))], 255)

    background = cv2.warpPerspective(ref_img, np.float32(th), (pano_w, pano_h))
    _, background_mask = cv2.threshold(cv2.cvtColor(background, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)
    # background_mask = cv2.fillPoly(background_mask, [np.int32(corners_dst)], 255)

    overlay_mask = cv2.bitwise_and(foreground_mask, background_mask)
    overlay_mask_3channel = cv2.merge([overlay_mask, overlay_mask, overlay_mask])
    no_overlay_mask = cv2.bitwise_not(overlay_mask)

    # cv2.imshow("Mask", imutils.resize(overlay_mask, width=1928))
    # cv2.waitKey()
    # Add the masked foreground and background.

    outImage = cv2.add(foreground, background, mask=no_overlay_mask)
    outImage = cv2.add(outImage,
                        cv2.addWeighted(cv2.bitwise_and(foreground, overlay_mask_3channel), 0.5,
                                        cv2.bitwise_and(background, overlay_mask_3channel), 0.5,
                                        0))
    # cv2.imshow("Stitched Img", imutils.resize(outImage, width=1928))

    # Add transform from the last stitching to the previous image transformations.
    if len(out_th_imgs) > 1:
        for idx in range(len(out_th_imgs)):
            out_th_imgs[idx] = th.dot(out_th_imgs[idx])
    else:
        out_th_imgs.append(th)
    out_th_imgs.append(th_img)

    if len(imgs) == 0:
        return outImage, out_th_imgs
    else:
        if ref_mask is not None:
            ref_mask = cv2.warpPerspective(ref_mask, np.float32(th), (pano_w, pano_h))
        return my_stitch([outImage, *imgs], ref_mask, out_th_imgs)


if __name__ == "__main__":
    intrinsics = CamIntrinsics(os.path.join("./camera", "Ezviz_C6N"))

    imgs, _ = undistort_images_from_files(["../imgs/pano/test_3_morning"], intrinsics)
    imgs[0], imgs[1] = imgs[1], imgs[0]
    ref_img = imgs[0]
    # ref_mask = create_mask(imgs[0])
    ref_mask = cv2.imread("base_rect_mask.jpg", cv2.IMREAD_GRAYSCALE)
    pano, transforms = my_stitch(imgs)

    #Test that image transform were computed correctly
    h, w = ref_img.shape[:2]
    corners_ref = [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]  # .reshape(-1, 1, 2)
    for t in transforms:
        corners_img = cv2.perspectiveTransform(np.float32([corners_ref]), t)
        pano = cv2.polylines(pano, [np.int32(corners_img)], True, (255, 255, 255), 3, cv2.LINE_AA)

    # pano = cv2.imread("base_rect_pano_plane.jpg")
    # ref_mask = cv2.imread("base_rect_mask.jpg", cv2.IMREAD_GRAYSCALE)
    # transforms = np.load("base_rect_pano_plane_transforms.npy")

    # Display image
    cv2.imshow("Pano Plane", imutils.resize(pano, width=1280))
    # cv2.imwrite("base_rect_pano_plane.jpg", pano)
    h, w = pano.shape[:2]
    cv2.imshow("base_rect_pano_plane_mask.jpg", imutils.resize(
        cv2.warpPerspective(ref_mask, np.float32(transforms[0]), (w, h)), width=1280))
    #np.save("base_rect_pano_plane_transforms", transforms)
    # pano = intrinsics.cylindrical_warp(pano)
    # # Display image
    # cv2.imshow("Pano Cylindrical Warp", imutils.resize(pano, width=1280))
    # cv2.imwrite("pano_cylindr.jpg", pano)


    cv2.waitKey(0)
    cv2.destroyAllWindows()