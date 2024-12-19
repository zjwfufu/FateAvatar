import os
import json
import math
import torch
import PIL.Image
import numpy as np

from typing import Union
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle


# ------------------------------------------------------------------------------- #

def load_rgb(path, img_res, bg_color='black', return_alpha=False):
    """
    Load and preprocess an RGBA or RGB image from the specified path.

    Args:
        path (str): The path to the image file.
        img_res (tuple): The target resolution (width, height) to resize the image.
        bg_color (str, optional): The background color to apply if the image has an alpha channel.
                                  Can be 'black' (default) or 'white'.
        return_alpha (bool, optional): Whether to return the alpha channel separately. Defaults to False.

    Returns:
        numpy.ndarray: A 3D numpy array of the image in shape (C, H, W) with pixel values in the range [0, 1].
        If `return_alpha` is True, also returns the alpha channel as a separate numpy array.
    
    Raises:
        ValueError: If an invalid `bg_color` is provided.
    """

    img = PIL.Image.open(path)
    img = img.resize(img_res)
    img = np.array(img)

    if img.shape[2] == 3:
        image = img[:, :, :3] / 255
    else:
        image = np.array(img, dtype=np.float32) / 255
        alpha = image[:, :, 3:4]
        if bg_color == 'white':
            image = image[:, :, :3] * alpha + (1 - alpha)
        elif bg_color == 'black': 
            image = image[:, :, :3] * alpha
        else:
            raise ValueError("Invalid Color!")
    
    image = image.transpose(2, 0, 1)

    if return_alpha:
        return image, alpha
    else:
        return image
    
# ------------------------------------------------------------------------------- #

def load_mask(path, img_res):
    """
    Load and preprocess a binary mask image from the specified path.

    Args:
        path (str): The path to the mask image file.
        img_res (tuple): The target resolution (width, height) to resize the mask image.

    Returns:
        numpy.ndarray: A 2D numpy array representing the mask with pixel values in the range [0, 1].
    """

    img = PIL.Image.open(path)
    img = img.convert('L')
    img = img.resize(img_res)
    img = np.array(img)

    object_mask = img / 255

    return object_mask
    
# ------------------------------------------------------------------------------- #

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data["image_paths"])
    
    def __getitem__(self):
        return NotImplementedError
    
    def collate_fn(self, batch_list):
        # this function is borrowed from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
    
# ------------------------------------------------------------------------------- #

class IMAvatarDataset(FaceDataset):
    def __init__(
            self,
            root_path:             str,
            mode:                  str,
            img_res:               list,
            hard_mask:             bool = False,
            use_mean_expression:   bool = False,
            use_var_expression:    bool = False,
            background_color:      str  = 'black',
            is_flashavatar:        bool = False,
        ):
        """
        Dataset for the DECA preprocessing pipeline

        Zheng et al, "I M Avatar: Implicit Morphable Head Avatars from Videos", CVPR 2022

        root_path:                 string to the specific subject directory with following layout
            <root_path>/
            |
            |---<subject>_test/
            |   |---...
            |
            |---<subject>_train/
                |---image/              # original images
                |    |---1.png
                |    |...
                |---mask/               # extracted mask containing the clothing region
                |    |---1.png
                |    |...
                |---matted/             # images with background and clothing removed
                |    |---1.png
                |    |...
                |---parsing/            # mask for mouth and neckhead, used in certain method
                |    |---1_mouth.png
                |    |---1_neckhead.png
                |    |...
                |---semantic/           # segmentation results, not used in here
                |    |---1.png
                |    |...
                |---semantic_color/     # segmentation visualization
                |    |---1.png
                |    |...
                |
                |---code.json           # not used in here
                |---flame_params.json   # the final json for train/test
                |---iris.json           # iris detection results, not used in here
                |---keypoint.json       # 2D keypoints, not used in here

        mode:                   whether it is the training set or the test set
        img_res:                a list containing height and width, e.g. [256, 256] or [512, 512]
        hard_mask:              whether to use boolean segmentation mask or not
        use_mean_expression:    if True, use mean expression of the training set as the canonical expression
        use_var_expression:     if True, blendshape regularization weight will depend on the variance of expression
                                (more regularization if variance is small in the training set.)
        background_color:       string to determine background color, black or white

        json file structure:
            frames:                 list of dictionaries, which are structured like:
                file_path:          relative path to image
                world_mat:          camera extrinsic matrix (world to camera). Camera rotation is actually the same for all frames,
                                    since the camera is fixed during capture.
                                    The FLAME head is centered at the origin, scaled by 4 times.
                expression:         50 dimension expression parameters
                pose:               15 dimension pose parameters
                flame_keypoints:    2D facial keypoints calculated from FLAME
            shape_params:           100 dimension FLAME shape parameters, shared by all scripts and testing frames of the subject
            intrinsics:             camera focal length fx, fy and the offsets of the principal point cx, cy. normalized.
        """
        self.root_path              = root_path
        self.mode                   = mode
        self.img_res                = img_res
        self.hard_mask              = hard_mask
        self.use_mean_expression    = use_mean_expression
        self.use_var_expression     = use_var_expression
        self.background_color       = background_color
        self.is_flashavatar         = is_flashavatar

        self.n_shape                = 100
        self.n_exp                  = 50
        self.optimize_tracking      = True
        self.type_name              = 'imavatar'
        
        self.data = {
            "image_paths": [],
            # camera extrinsics
            "world_mats": [],
            # FLAME expression and pose parameters
            "expressions": [],
            "flame_pose": [],
            # saving image names and subdirectories
            "img_name": [],
        }

        fix_cam_rot = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).astype(np.float32)
        self.fix_cam_rot = torch.from_numpy(fix_cam_rot)

        json_name = 'flame_params.json'
        instance_dir = os.path.join(root_path, f"{os.path.basename(root_path)}_{mode}")

        assert os.path.exists(instance_dir), f"Data directory {instance_dir} is empty"

        cam_file = f'{instance_dir}/{json_name}'
        with open(cam_file, 'r') as f:
            camera_dict = json.load(f)

        for frame in camera_dict['frames']:
            world_mat = np.array(frame['world_mat']).astype(np.float32)
            world_mat[:, 3] /= 4

            self.data["world_mats"].append(world_mat)
            self.data["expressions"].append(np.array(frame['expression']).astype(np.float32))
            self.data["flame_pose"].append(np.array(frame['pose']).astype(np.float32))

            image_path = os.path.join(instance_dir, f"{frame['file_path']}.png")
            self.data["image_paths"].append(image_path.replace('image', 'matted'))      # we use parsed images for fair settings
            self.data["img_name"].append(int(frame["file_path"].split('/')[-1]))

        self.gt_dir = instance_dir
        self.shape_params = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0)
        

        self.data["expressions"] = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["flame_pose"] = torch.from_numpy(np.stack(self.data["flame_pose"], 0))
        self.data["world_mats"] = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()

        # construct intrinsic matrix
        intrinsics = np.zeros((4, 4))
        focal_cxcy = camera_dict['intrinsics']
        intrinsics[0, 0] = focal_cxcy[0] * 2
        intrinsics[1, 1] = focal_cxcy[1] * 2
        intrinsics[0, 2] = (focal_cxcy[2] * 2 - 1.0) * -1
        intrinsics[1, 2] = (focal_cxcy[3] * 2 - 1.0) * -1
        intrinsics[3, 2] = 1.
        intrinsics[2, 3] = 1.
        self.intrinsics = intrinsics
        self.intrinsics = torch.from_numpy(self.intrinsics).float()

        # calculate FOV
        fx = focal_cxcy[0] * -1
        fy = focal_cxcy[1]
        cx = focal_cxcy[2]
        cy = focal_cxcy[3]
        self.fovx = 2 * math.atan2(cx, fx)
        self.fovy = 2 * math.atan2(cy, fy)

        # from whatever camera convention to another whatever camera convention
        if intrinsics[0, 0] < 0:
            intrinsics[:, 0] *= -1
            self.data["world_mats"][:, 0, :] *= -1
        self.data["world_mats"][:, :3, 2] *= -1
        self.data["world_mats"][:, 2, 3] *= -1

        self.data["world_mats"][:, 0, :] *= -1
        self.data["world_mats"][:, 1, :] *= -1

        if use_mean_expression:
            self.mean_expression = torch.mean(self.data["expressions"], 0, keepdim=True)
        else:
            self.mean_expression = torch.zeros_like(self.data["expressions"][[0], :])
        if use_var_expression:
            self.var_expression = torch.var(self.data["expressions"], 0, keepdim=True)
        else:
            self.var_expression = None

    def __getitem__(self, idx):

        sample = {
            "idx":          torch.LongTensor([idx]),
            "img_name":     torch.LongTensor([self.data["img_name"][idx]]),
            "intrinsics":   self.intrinsics,
            "fovx":         self.fovx,
            "fovy":         self.fovy,
            "expression":   self.data["expressions"][idx],
            "flame_pose":   self.data["flame_pose"][idx],
            "cam_pose":     self.data["world_mats"][idx],
        }

        ground_truth = {}

        rgb, object_mask = load_rgb(
            self.data["image_paths"][idx],
            self.img_res,
            bg_color        = self.background_color,
            return_alpha    = True
        )

        rgb = torch.from_numpy(rgb).float()
        object_mask = torch.from_numpy(object_mask).float()

        ground_truth = {
            'rgb': rgb,
            'object_mask': object_mask
        }

        if self.is_flashavatar:
            mouth_mask_path = self.data["image_paths"][idx].replace('matted', 'parsing').replace('.png', '_mouth.png')
            mouth_mask      = torch.from_numpy(load_mask(mouth_mask_path, self.img_res)).unsqueeze(0)

            ground_truth.update({'mouth_mask': mouth_mask})

        return idx, sample, ground_truth

# ------------------------------------------------------------------------------- #

class InstaDataset(FaceDataset):
    def __init__(
        self,
        root_path:             str,
        mode:                  str,
        img_res:               list,
        hard_mask:             bool = False,
        use_mean_expression:   bool = False,
        use_var_expression:    bool = False,
        background_color:      str  = 'black',
        is_flashavatar:        bool = False,
    ):
        """
        Dataset for the INSTA preprocessing pipeline

        Zielonka et al, "Instant Volumetric Head Avatars", CVPR 2023

        root_path:                  string to the specific subject directory with following layout
            <root_path>/
            |---alpha/                  # extracted mask containing the clothing region
            |   |---00000.png
            |   |...
            |---checkpoint/             # dumped coefficients meta data
            |   |---00000.frame     
            |   |...
            |---depth/                  # estimated depth map, not used here
            |   |---00000.png
            |   |...
            |---flame/                  # FLAME coefficients
            |   |---exp/                # expression param
            |   |   |---00000.txt
            |   |   |...
            |   |---eyelids/            # coeff for additional eyelid blendshape, not used here
            |   |   |---00000.txt
            |   |   |...
            |   |---eyes/               # eye pose, 6d rotation
            |   |   |---00000.txt
            |   |   |...
            |   |---jaw/                # jaw pose, 6d rotation
            |   |   |---00000.txt
            |   |   |...
            |   |---shape/              # shape param
            |       |---00000.txt
            |       |...
            |---images/                 # images with background and clothing removed 
            |   |---00000.png
            |   |...
            |---imgs/                   # original images
            |   |---00000.png
            |   |...
            |---matted/                 # images with clothing removed
            |   |---00000.png
            |   |...
            |---meshes/                 # FLAME mesh in each frame for visualization
            |   |---00000.obj
            |   |...
            |---parsing/                # mask for mouth and neckhead, used in certain method
            |   |---00000_mouth.png
            |   |---00000_neckhead.png
            |   |...
            |---seg_mask/               # segmentation results, not used in here
            |   |---00000.png
            |   |...
            |
            |---canonical.obj
            |---transforms.json         # the final transforms.json
            |---transforms_train.json   # the final transforms.json for training
            |---transforms_val.json     # the final transforms.json for validation
            |---transforms_test.json    # the final transforms.json for testing

                json file structure:
                    frames:     A list of dictionaries, where each dictionary contains data for a frame:
                        depth_path:         The file path to the depth image.
                        exp_path:           The file path to the FLAME expression parameters.
                        file_path:          The file path to the RGB image.
                        mesh_path:          The file path to the 3D mesh.
                        seg_mask_path:      The file path to the semantic segmentation mask.
                        transform_matrix:   A 4x4 camera extrinsic matrix that describes the transformation from world coordinates to camera coordinates.

                    camera_angle_x:         The camera's field of view (FOV) along the x-axis, in radians.
                    camera_angle_y:         The camera's field of view (FOV) along the y-axis, in radians.
                    cx:                     The x-coordinate offset of the camera's principal point (in pixels).
                    cy:                     The y-coordinate offset of the camera's principal point (in pixels).
                    fl_x:                   The camera's focal length along the x-axis (in pixels).
                    fl_y:                   The camera's focal length along the y-axis (in pixels).

                    h:                      The image height (in pixels).
                    integer_depth_scale:    The scale factor for converting depth values.
                    w:                      The image width (in pixels).
                    x_fov:                  The field of view in the x direction (in degrees).
                    y_fov:                  The field of view in the y direction (in degrees).
            
        mode:                   whether it is the training set or the test set
        img_res:                a list containing height and width, e.g. [256, 256] or [512, 512]
        hard_mask:              whether to use boolean segmentation mask or not
        use_mean_expression:    if True, use mean expression of the training set as the canonical expression
        use_var_expression:     if True, blendshape regularization weight will depend on the variance of expression
                                (more regularization if variance is small in the training set.)
        background_color:       string to determine background color, black or white. 
        """
        self.root_path              = root_path
        self.mode                   = mode
        self.img_res                = img_res
        self.hard_mask              = hard_mask
        self.use_mean_expression    = use_mean_expression
        self.use_var_expression     = use_var_expression
        self.background_color       = background_color
        self.is_flashavatar         = is_flashavatar

        self.n_shape                = 300
        self.n_exp                  = 100
        self.optimize_tracking      = False
        self.type_name              = 'insta'

        # if True, we rotate camera for modeling head rotation, otherwise use FLAME root pose
        # In original insta, it uses camera rotation.
        self.rot_camera = True

        fix_cam_rot = np.array([
            [1, 0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ]).astype(np.float32)
        self.fix_cam_rot = torch.from_numpy(fix_cam_rot)

        if not self.rot_camera:
            global_rot_mat = []

        self.data = {
            "image_paths":  [],
            "mask_paths":   [],
            # camera extrinsics
            "world_mats":   [],
            # FLAME expression and pose parameters
            "expressions":  [],
            "flame_pose":   [],
            "jaw_pose_6d":  [],
            "eyes_pose_6d": [],
            # saving image names and subdirectories
            "img_name":     [],
        }

        transform_file = 'transforms_train.json' if self.mode == 'train' else 'transforms_test.json'
        transform_path = os.path.join(self.root_path, transform_file)
        with open(transform_path, 'r') as f:
            transform = json.load(f)

        transform['frames'].sort(key=lambda x: int(x['file_path'].split('/')[-1].split('.')[0]))
        num_frames = len(transform['frames'])

        # load parameters
        for idx, frame in enumerate(transform['frames']):
            c2w = np.array(frame['transform_matrix']).astype(np.float32)
            world_mat = np.linalg.inv(c2w)

            if self.rot_camera:
                world_mat[:3, :3] = np.transpose(world_mat[:3, :3])
                self.data["world_mats"].append(world_mat[:3])
            else:
                rot_mat         = world_mat[:3, :3].copy()
                rot_mat[1, :]   *= -1
                rot_mat[2, :]   *= -1
                global_rot_mat.append(rot_mat)
                self.data["world_mats"].append(np.concatenate([fix_cam_rot, world_mat[:3,3][...,None]], axis=1))

            if idx == 0:
                shape = np.loadtxt(os.path.join(self.root_path, frame['exp_path'].replace('exp', 'shape')))

            exp     = np.loadtxt(os.path.join(self.root_path, frame['exp_path']))
            eyes    = np.loadtxt(os.path.join(self.root_path, frame['exp_path'].replace('exp', 'eyes')))
            jaw     = np.loadtxt(os.path.join(self.root_path, frame['exp_path'].replace('exp', 'jaw')))

            self.data["expressions"].append(exp.astype(np.float32))
            self.data["eyes_pose_6d"].append(eyes.astype(np.float32))
            self.data["jaw_pose_6d"].append(jaw.astype(np.float32))
            self.data["image_paths"].append(os.path.join(self.root_path, frame['file_path']))
            self.data["mask_paths"].append(os.path.join(self.root_path, frame['file_path'].replace('images', 'alpha')))
            self.data["img_name"].append(frame['file_path'].split('/')[-1])
            
        # stack list
        I = matrix_to_axis_angle(torch.cat([torch.eye(3)[None]] * num_frames, dim=0))
        self.shape_params = torch.tensor(shape).float().unsqueeze(0)

        self.data["expressions"]    = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["eyes_pose_6d"]   = torch.from_numpy(np.stack(self.data["eyes_pose_6d"], 0))
        self.data["jaw_pose_6d"]    = torch.from_numpy(np.stack(self.data["jaw_pose_6d"], 0))

        l_eye_pose_6d               = self.data["eyes_pose_6d"][:, 6:]
        r_eye_pose_6d               = self.data["eyes_pose_6d"][:, :6]
        jaw_pose_6d                 = self.data["jaw_pose_6d"]

        l_eye_pose                  = matrix_to_axis_angle(rotation_6d_to_matrix(l_eye_pose_6d))
        r_eye_pose                  = matrix_to_axis_angle(rotation_6d_to_matrix(r_eye_pose_6d))
        jaw_pose                    = matrix_to_axis_angle(rotation_6d_to_matrix(jaw_pose_6d))
        neck_pose                   = I.clone()

        if self.rot_camera:
            rot_pose                = I.clone()
        else:
            global_rot_mat          = torch.from_numpy(np.stack(global_rot_mat, 0)).float()
            rot_pose                = matrix_to_axis_angle(global_rot_mat)

        self.data["flame_pose"]     = torch.cat([rot_pose, neck_pose, jaw_pose, l_eye_pose, r_eye_pose], dim=1)
        self.data["world_mats"]     = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()

        focal_cxcy = [
            0.5 * transform['fl_x'] / transform['cx'],
            0.5 * transform['fl_y'] / transform['cy'],
            transform['cx'] / transform['w'],
            transform['cy'] / transform['h']
        ]

        # construct intrinsic matrix
        intrinsics          = np.zeros((4, 4))
        intrinsics[0, 0]    = focal_cxcy[0] * 2
        intrinsics[1, 1]    = focal_cxcy[1] * 2
        intrinsics[0, 2]    = (focal_cxcy[2] * 2 - 1.0) * -1
        intrinsics[1, 2]    = (focal_cxcy[3] * 2 - 1.0) * -1
        intrinsics[3, 2]    = 1.
        intrinsics[2, 3]    = 1.
        self.intrinsics     = intrinsics
        self.intrinsics     = torch.from_numpy(self.intrinsics).float()

        # calculate FOV
        self.fovx = 2 * math.atan2(focal_cxcy[2], focal_cxcy[0])
        self.fovy = 2 * math.atan2(focal_cxcy[3], focal_cxcy[1])

        if use_mean_expression:
            self.mean_expression = torch.mean(self.data["expressions"], 0, keepdim=True)
        else:
            self.mean_expression = torch.zeros_like(self.data["expressions"][[0], :])
        if use_var_expression:
            self.var_expression = torch.var(self.data["expressions"], 0, keepdim=True)
        else:
            self.var_expression = None

    def __getitem__(self, idx):

        sample = {
            "idx":          torch.LongTensor([idx]),
            "img_name":     self.data["img_name"][idx],
            "intrinsics":   self.intrinsics,
            "fovx":         self.fovx,
            "fovy":         self.fovy,
            "expression":   self.data["expressions"][idx],
            "flame_pose":   self.data["flame_pose"][idx],
            "cam_pose":     self.data["world_mats"][idx],
        }

        ground_truth = {}

        rgb, object_mask = load_rgb(
            self.data["image_paths"][idx],
            self.img_res,
            bg_color        = self.background_color,
            return_alpha    = True
        )

        rgb         = torch.from_numpy(rgb).float()
        object_mask = torch.from_numpy(object_mask).float()

        ground_truth = {
            'rgb':          rgb,
            'object_mask':  object_mask
        }

        if self.is_flashavatar:
            mouth_mask_path = self.data["image_paths"][idx].replace('images', 'parsing').replace('.png', '_mouth.png')
            mouth_mask      = torch.from_numpy(load_mask(mouth_mask_path, self.img_res)).unsqueeze(0)

            ground_truth.update({'mouth_mask': mouth_mask})

        return idx, sample, ground_truth
    
# ------------------------------------------------------------------------------- #

DatasetClass = Union[
    IMAvatarDataset, InstaDataset
]