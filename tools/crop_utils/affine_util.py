import cv2
import numpy as np
from math import cos, sin, atan2, asin, sqrt

def eg3dcamparams(R_in):
    camera_dist = 2.7
    intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
    # assume inputs are rotation matrices for world2cam projection
    R = np.array(R_in).astype(np.float32).reshape(4,4)
    # add camera translation
    t = np.eye(4, dtype=np.float32)
    t[2, 3] = - camera_dist

    # convert to OpenCV camera
    convert = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)

    # world2cam -> cam2world
    P = convert @ t @ R
    cam2world = np.linalg.inv(P)

    # add intrinsics
    label_new = np.concatenate([cam2world.reshape(16), intrinsics.reshape(9)], -1)
    return label_new

def get_crop_bound(lm, method="ffhq"):
    if len(lm) == 106:
        left_e = lm[104]
        right_e = lm[105]
        nose = lm[49]
        left_m = lm[84]
        right_m = lm[90]
        center = (lm[1] + lm[31]) * 0.5
    elif len(lm) == 68:
        left_e = np.mean(lm[36:42], axis=0)
        right_e = np.mean(lm[42:48], axis=0)
        nose = lm[33]
        left_m = lm[48]
        right_m = lm[54]
        center = (lm[0] + lm[16]) * 0.5
    else:
        raise ValueError(f"Unknown type of keypoints with a length of {len(lm)}")

    if method == "ffhq":
        eye_to_eye = right_e - left_e
        eye_avg = (left_e + right_e) * 0.5
        mouth_avg = (left_m + right_m) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
    elif method == "default":
        eye_to_eye = right_e - left_e
        eye_avg = (left_e + right_e) * 0.5
        eye_to_nose = nose - eye_avg
        x = eye_to_eye.copy()
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.4, np.hypot(*eye_to_nose) * 2.75)
        y = np.flipud(x) * [-1, 1]
        c = center
    else:
        raise ValueError('%s crop method not supported yet.' % method)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    return quad.astype(np.float32), c, x, y

def crop_image(img, mat, crop_w, crop_h, upsample=1, borderMode=cv2.BORDER_CONSTANT):
    crop_size = (crop_w, crop_h)
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 
    return crop_img

def crop_final(
    img, 
    size=512, 
    quad=None,
    top_expand=0.1, 
    left_expand=0.05, 
    bottom_expand=0.0, 
    right_expand=0.05, 
    blur_kernel=None,
    borderMode=cv2.BORDER_REFLECT,
    upsample=2,
    min_size=256,
):  

    orig_size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))

    if min_size is not None and orig_size < min_size:
        return None

    crop_w = int(size * (1 + left_expand + right_expand))
    crop_h = int(size * (1 + top_expand + bottom_expand))
    crop_size = (crop_w, crop_h)
    
    top = int(size * top_expand)
    left = int(size * left_expand)
    size -= 1
    bound = np.array([[left, top], [left, top + size], [left + size, top + size], [left + size, top]],
                        dtype=np.float32)

    mat = cv2.getAffineTransform(quad[:3], bound[:3])
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 

    empty = np.ones_like(img) * 255
    crop_mask = cv2.warpAffine(empty, mat, crop_size)



    if True:
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1 if blur_kernel is None else blur_kernel
        downsample_size = (crop_w//8, crop_h//8)
        
        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2),(mask_kernel,mask_kernel)) / 255.0
            blur_mask = blur_mask[...,np.newaxis]#.astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)
    
    return crop_img

def find_center_bbox(roi_box_lst, w, h):
    bboxes = np.array(roi_box_lst)
    dx = 0.5*(bboxes[:,0] + bboxes[:,2]) - 0.5*(w-1)
    dy = 0.5*(bboxes[:,1] + bboxes[:,3]) - 0.5*(h-1)
    dist = np.stack([dx,dy],1)
    return np.argmin(np.linalg.norm(dist, axis=1))


def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z