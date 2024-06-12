import cv2
import numpy as np


def sort_corner_order(quadrangle: np.ndarray):
    assert quadrangle.shape == (4, 1, 2), f'Invalid quadrangle shape: {quadrangle.shape}'

    quadrangle = quadrangle.squeeze(1)
    moments = cv2.moments(quadrangle)
    mcx = round(moments['m10'] / moments['m00'])  # mass center x
    mcy = round(moments['m01'] / moments['m00'])  # mass center y
    keypoints = np.zeros((4, 2), np.int32)
    for point in quadrangle:
        if point[0] < mcx and point[1] < mcy:
            keypoints[0] = point
        elif point[0] < mcx and point[1] > mcy:
            keypoints[1] = point
        elif point[0] > mcx and point[1] > mcy:
            keypoints[2] = point
        elif point[0] > mcx and point[1] < mcy:
            keypoints[3] = point
    return keypoints


def get_keypoints(masks: np.ndarray, morph_ksize=21, contour_thres=0.02, poly_thres=0.03):
    # Perform morphological transformation
    masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize)))
    # Find contours (+remove noise)
    contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = [contour for contour in contours
                if cv2.contourArea(contour) > (masks.shape[0] * masks.shape[1] * contour_thres)]
    # Approximate quadrangles (+remove noise)
    quadrangles = [cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * poly_thres, True) for contour in contours]
    quadrangles = [quad for quad in quadrangles if quad.shape == (4, 1, 2)]

    if len(quadrangles) == 1:
        keypoints = sort_corner_order(quadrangles[0])
        return keypoints
    else:
        return None


def align_idcard(img: np.ndarray, keypoints: np.ndarray):
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)
    idcard_ratio = np.array((86, 54))
    dsize_factor = round(np.sqrt(cv2.contourArea(np.expand_dims(keypoints, 1))) / idcard_ratio[0])

    dsize = idcard_ratio * dsize_factor  # idcard size unit: mm
    dst = np.array(((0, 0), (0, dsize[1]), dsize, (dsize[0], 0)), np.float32)

    M = cv2.getPerspectiveTransform(keypoints.astype(np.float32), dst)
    img = cv2.warpPerspective(img, M, dsize)
    return img