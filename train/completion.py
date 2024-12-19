import  glob
import  os
import  sys
import  math
import  tqdm
import  imageio
import  numpy as np
import  pickle
import  cv2
import  yaml
import  copy
import  json
import  PIL.Image
import  PIL.ImageChops
import  PIL.ImageOps
import  shutil
import  re

import  warnings
warnings.filterwarnings("ignore", category=UserWarning)

import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
import  torchvision
import  torchvision.transforms as transforms

from    typing import Type
from    model import ModelClass

from    tools.util         import EasyDict, load_to_gpu
from    tools.eg3d_utils.camera_eg3d import LookAtPoseSampler

from    pytorch3d.transforms import matrix_to_axis_angle

from    train.loader   import Loader

# ------------------------------------------------------------------------------- #

class PseudoGenerator(Loader):
    def __init__(
            self,
            name:              str,
            cfg:               EasyDict,
            model:             Type[ModelClass],
            device:            torch.device,
            workspace:         str,
            use_checkpoints:   str='latest',
            max_keep_ckpt:     int=2,
            bg_color:          str='white'
        ):
        super().__init__(name, cfg, model, device,
                         workspace, use_checkpoints, max_keep_ckpt)
        
        """
        The main goal of this class is to align images rendered using GS-based methods with
        those rendered by the pretrained 3D-aware GAN, which is a little bit tricky.
        """

        self.enhenced   = False
        self.bg_color   = bg_color

        self.dlib_threshold     = 1.0

        self.rotate_type        = 'camera'     # also try to be 'flame', but not good

        self.pretrained_type    = 'spherehead' # or 'panohead', but spherehead is more stable

        self.pti_w_step         = 200
        self.pti_finetune_step  = 200

        self.rescale_scene      = True # To make the 3D-GAN output more complete, the nerf-scene can be scaled. However, this may result in a loss of image quality.
        self.rescale_factor     = 0.5

        # calculate flame rot joint position
        from flame.lbs import vertices2joints

        try:
            verts_cano, _, _ = self.model.flame.forward_with_delta_blendshape(
                expression_params   = self.model.flame.canonical_exp,
                full_pose           = self.model.flame.canonical_pose,
                delta_shapedirs     = self.model.delta_shapedirs if self.cfg.model.delta_blendshape else None,
                delta_posedirs      = self.model.delta_posedirs if self.cfg.model.delta_blendshape else None,
                delta_vertex        = self.model.delta_vertex if self.cfg.model.delta_vertex else None
            )
        except:
            verts_cano, _, _ = self.model.flame(
                expression_params   = self.model.flame.canonical_exp,
                full_pose           = self.model.flame.canonical_pose,
            )

        J = vertices2joints(self.model.flame.J_regressor, verts_cano)
        J_rot = J[0, 0]

        # carefully tune the last number...
        gs_camera_lookat_point = torch.tensor([0, 0, -0.02]).to(self.device)

        self.J = J
        self.gs_camera_lookat_point = gs_camera_lookat_point
        self.gs_camera_radius = self.cfg.camera_translation[-1]

        self.register_media_save()
        self.register_weight_path()

    def register_media_save(self):

        self.media_save_path = {
            "aug_workspace": {
                "folder": os.path.join(self.workspace, "augmentation")
            },
            "video": {
                "folder": os.path.join(self.workspace, "augmentation", "video")
            },
            "render_novel_view": {
                "folder": os.path.join(self.workspace, "augmentation", "novel_view")
            },
            "affine_transform": {
                "folder": os.path.join(self.workspace, "augmentation", "crop_images")
            },
            "inject_prior": {
                "folder": os.path.join(self.workspace, "augmentation", "crop_images_sr")
            },
            "run_pti":  {
                "folder": os.path.join(self.workspace, "augmentation", "gan_pti")
            },
            "inverse_transform":  {
                "folder": os.path.join(self.workspace, "augmentation", "paste_back")
            },
            "heatmap_check":    {
                "folder": os.path.join(self.workspace, "augmentation", "heatmap_check")
            }
        }

    def register_weight_path(self):

        self.weight_path = {
            "dlib": "./weights/shape_predictor_68_face_landmarks.dat",
            "gfpgan": "./weights/GFPGANv1.3.pth",
            "3d-gan": {
                "spherehead": "./weights/spherehead-ckpt-025000.pkl",
                "panohead": "./weights/easy-khair-180-gpc0.8-trans10-025000.pkl"
            },
            "bisenet": "./weights/79999_iter.pth",
            "modnet": "./weights/modnet_webcam_portrait_matting.ckpt",
        }

        for key, path in self.weight_path.items():
            if key == "3d-gan":
                if self.pretrained_type not in path:
                    raise ValueError(f"Invalid pretrain type '{self.pretrained_type}' for 3d-gan")
                specific_path = path[self.pretrained_type]
                if not os.path.exists(specific_path):
                    raise FileNotFoundError(f"Weight file for '{key}' ({specific_path}) is missing")
            else:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Weight file for '{key}' ({path}) is missing.")
        
    @torch.no_grad()
    def render_novel_view(self, orbit_frames=40, ele_list=[0]):
        """
        Render novel view images from pretrained head avatar.
        """

        save_path = self.media_save_path["render_novel_view"]["folder"]
        os.makedirs(save_path, exist_ok=True)

        self.log('++> Render novel view images...')

        self.model.eval()

        total_len   = orbit_frames * len(ele_list)
        pbar        = tqdm.tqdm(total=total_len, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        input_data      = {}
        ground_truth    = {}

        input_data['expression'] = self.model.canonical_expression
        input_data['flame_pose'] = self.model.flame.canonical_pose
        
        input_data['fovx']  = [self.cfg.camera_fovx]
        input_data['fovy']  = [self.cfg.camera_fovy]

        #### insert a ugly fix ####
        if self.cfg.camera_rotation[1, 1] == 1.:
            self.cfg.camera_rotation[1, 1] = -1
            self.cfg.camera_rotation[2, 2] = -1

        input_data['cam_pose'] = torch.cat((self.cfg.camera_rotation, self.cfg.camera_translation[..., None]), dim=1)
        # erase translation in x, y
        input_data['cam_pose'][0, 3] = 0
        input_data['cam_pose'][1, 3] = 0
        input_data['cam_pose'] = input_data['cam_pose'][None, ...]

        results_cam_pose    = {}

        all_render_np   = []
        save_path_videos    = os.path.join(self.media_save_path["video"]["folder"], 'novel_view.mp4')
        os.makedirs(os.path.dirname(save_path_videos), exist_ok=True)

        for round, ele in enumerate(ele_list):
            for frame_idx in range(1, orbit_frames + 1):
                
                cam2world_pose  = LookAtPoseSampler.sample(
                    math.pi / 2 + 2 * math.pi * (frame_idx - 1) / orbit_frames,
                    math.pi / 2 - ele,
                    self.gs_camera_lookat_point,
                    radius  = self.gs_camera_radius,   # R
                    device  = self.device
                )

                # type-I: rotate camera
                if self.rotate_type == 'camera':
                    # input_data['cam_pose'][:, :3, :3] = cam2world_pose[:, :3, :3]

                    world2cam = torch.linalg.inv(cam2world_pose)
                    input_data['cam_pose'][:, :3, :4] = world2cam[:, :3, :4]

                # type-II: rotate head
                elif self.rotate_type == 'flame':
                    R = cam2world_pose[:, :3, :3]
                    R[:, 1] = R[:, 1] * -1
                    R[:, 2] = R[:, 2] * -1
                    rot_vec = matrix_to_axis_angle(R)
                    input_data["flame_pose"][:, :3] = rot_vec
                else:
                    raise

                load_to_gpu(input_data, ground_truth, self.device)
                output_data     = self.model(input_data)
                render_image    = output_data['rgb_image']
                render_np       = render_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                all_render_np.append((render_np * 255.).astype('uint8'))
                render_plot     = render_image[0].detach().cpu()
                image_name      = os.path.join(save_path, f'{frame_idx + round * orbit_frames:04d}.png')
                torchvision.utils.save_image(render_plot, image_name, normalize=True, value_range=(0, 1))

                results_cam_pose[image_name]    = cam2world_pose.squeeze().cpu().numpy()

                pbar.update(1)

        all_render_np = np.stack(all_render_np, axis=0)
        imageio.mimwrite(save_path_videos, all_render_np, fps=25, quality=10)

        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'c2w.pkl'), 'wb') as f:
            pickle.dump(results_cam_pose, f)

        pbar.close()
        self.log_file_only(pbar)

        self.log('++> Render novel view finished.')

    def detect_dlib_kps(self):
        """
        Run dlib keypoints detection for further crop and filter out invalid images
        """
        import dlib
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']

        detector    = dlib.get_frontal_face_detector()
        predictor   = dlib.shape_predictor(self.weight_path["dlib"])

        self.log('++> Run dlib keypoints detection.')

        # load images
        img_dir     = self.media_save_path["render_novel_view"]["folder"]
        list_dir    = os.listdir(img_dir)

        # new dict for keypoints
        landmarks = {}
        pbar      = tqdm.tqdm(total=len(list_dir), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for img_name in list_dir:
            pbar.update(1)
            _, extension = os.path.splitext(img_name)
            if extension not in image_extensions:
                continue
            
            img_path    = os.path.join(img_dir, img_name)
            image       = cv2.imread(img_path)
            # gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect face
            rects = detector(gray, 1)
            dets, scores, idx = detector.run(gray, 1)
            self.log(f"{img_path}: {scores}")

            if len(scores) == 0 or scores[0] < self.dlib_threshold:
                continue

            for (i, rect) in enumerate(rects):
                # get keypoints
                shape = predictor(gray, rect)
                # save kps to the dict
                landmarks[img_path] = [np.array([p.x, p.y]) for p in shape.parts()]

        # save the data.pkl pickle
        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'dlib_kps.pkl'), 'wb') as f:
            pickle.dump(landmarks, f)

        pbar.close()
        self.log_file_only(pbar)
        self.log('++> Dlib keypoints detection finished.')

    @torch.no_grad()
    def execute_affine_transform(self):
        """
        Align images with GAN preprocessing
        """
        root_dir = os.getcwd()

        if root_dir in sys.path:
            sys.path.remove(root_dir)

        TDDFA_LIB_PATH  = os.path.abspath(os.path.join(os.getcwd(), 'submodules/3DDFA_V2'))
        sys.path.insert(0, TDDFA_LIB_PATH)

        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA
        from tools.crop_utils.affine_util import (get_crop_bound, crop_image,
                                    find_center_bbox, crop_final,
                                    P2sRt, matrix2angle,
                                    eg3dcamparams)
        
        sys.path.insert(0, root_dir)

        self.log('++> Run Affine Align.')

        #----- load 3ddfa config -----#
        cur_dir = os.getcwd()
        os.chdir('./submodules/3DDFA_V2')
        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        
        tddfa = TDDFA(gpu_mode='gpu', **cfg)
        face_boxes = FaceBoxes()
        
        os.chdir(cur_dir)
        #------------------------------#

        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'dlib_kps.pkl'), "rb") as f:
            inputs = pickle.load(f, encoding="latin1").items()

        pbar = tqdm.tqdm(total=len(inputs), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        size = 512
        results_quad = {}
        results_meta = {}
        results_orig_quad = {}
        for i, item in enumerate(inputs):
            pbar.update(1)

            # get initial cropping box (quad) using landmarks
            img_path, landmarks = item
            img_path = img_path
            img_orig = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            if img_orig is None:
                print(f'Cannot load image')
                continue
            quad, quad_c, quad_x, quad_y = get_crop_bound(landmarks)

            results_orig_quad[img_path] = copy.deepcopy(quad)

            skip = False
            for iteration in range(1):
                bound = np.array([[0, 0], [0, size-1], [size-1, size-1], [size-1, 0]], dtype=np.float32)
                mat = cv2.getAffineTransform(quad[:3], bound[:3])
                img = crop_image(img_orig, mat, size, size)
                # img = img_orig
                h, w = img.shape[:2]

                # Detect faces, get 3DMM params and roi boxes
                boxes = face_boxes(img)
                # boxes = face_boxes(img_orig)
                if len(boxes) == 0:
                    print(f'No face detected')
                    skip = True
                    break

                param_lst, roi_box_lst = tddfa(img, boxes)
                # param_lst, roi_box_lst = tddfa(img_orig, boxes)
                box_idx = find_center_bbox(roi_box_lst, w, h)

                param = param_lst[box_idx]
                P = param[:12].reshape(3, -1)  # camera matrix
                s_relative, R, t3d = P2sRt(P)

                pose = matrix2angle(R)
                pose = [p * 180 / np.pi for p in pose]

                # Adjust z-translation in object space
                R_ = param[:12].reshape(3, -1)[:, :3]
                u = tddfa.bfm.u.reshape(3, -1, order='F')
                trans_z = np.array([ 0, 0, 0.5*u[2].mean() ]) # Adjust the object center
                trans = np.matmul(R_, trans_z.reshape(3,1))
                t3d += trans.reshape(3)

                ''' Camera extrinsic estimation for GAN training '''
                # Normalize P to fit in the original image (before 3DDFA cropping)
                sx, sy, ex, ey = roi_box_lst[0]
                scale_x = (ex - sx) / tddfa.size
                scale_y = (ey - sy) / tddfa.size
                t3d[0] = (t3d[0]-1) * scale_x + sx
                t3d[1] = (tddfa.size-t3d[1]) * scale_y + sy
                t3d[0] = (t3d[0] - 0.5*(w-1)) / (0.5*(w-1)) # Normalize to [-1,1]
                t3d[1] = (t3d[1] - 0.5*(h-1)) / (0.5*(h-1)) # Normalize to [-1,1], y is flipped for image space
                t3d[1] *= -1
                t3d[2] = 0 # orthogonal camera is agnostic to Z (the model always outputs 66.67)

                s_relative = s_relative * 2000
                scale_x = (ex - sx) / (w-1)
                scale_y = (ey - sy) / (h-1)
                s = (scale_x + scale_y) / 2 * s_relative
                # print(f"[{iteration}] s={s} t3d={t3d}")

                if s < 0.7 or s > 1.3:
                    print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} s={s}")
                    skip = True
                    break
                if abs(pose[0]) > 90 or abs(pose[1]) > 80 or abs(pose[2]) > 50:
                    print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} pose={pose}")
                    skip = True
                    break
                if abs(t3d[0]) > 1. or abs(t3d[1]) > 1.:
                    print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} pose={pose} t3d={t3d}")
                    skip = True
                    break

                quad_c = quad_c + quad_x * t3d[0]
                quad_c = quad_c - quad_y * t3d[1]
                quad_x = quad_x * s
                quad_y = quad_y * s
                c, x, y = quad_c, quad_x, quad_y
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)
                

            if skip:
                continue

            # final projection matrix
            s = 1
            t3d = 0 * t3d
            R[:,:3] = R[:,:3] * s
            P = np.concatenate([R,t3d[:,None]],1)
            P = np.concatenate([P, np.array([[0,0,0,1.]])],0)

            # Save cropped images

            cropped_img = crop_final(img_orig, size=size, quad=quad)
            # cropped_img = crop_final(img_orig, size=size, quad=quad, top_expand=0.0, left_expand=0.0, bottom_expand=0.0, right_expand=0.0)

            out_dir = self.media_save_path["affine_transform"]["folder"]

            if cropped_img is not None:
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), cropped_img)

                results_meta[img_path] = eg3dcamparams(P.flatten())
                results_quad[img_path] = quad

        pbar.close()
        self.log_file_only(pbar)

        self.log('++> Align Finished.')

        # if self.bg_color == 'black':
        #     # turn the background into white for inversion
        #     self.retrieve_image_mask_modnet(input_path = out_dir, output_path = out_dir.replace('multi_view_crop', 'multi_view_crop_mask'))

        #     for filename in os.listdir(out_dir):
        #         if filename.endswith('.json'):
        #             continue

        #         image = cv2.imread(os.path.join(out_dir, filename)).astype(np.float32) # [0, 255]
        #         mask  = cv2.imread(os.path.join(out_dir.replace('multi_view_crop', 'multi_view_crop_mask'), filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)  # [0, 255]

        #         image /= 255.
        #         mask  /= 255.

        #         image = image * mask[..., None] + (1 - mask[..., None])

        #         cv2.imwrite(os.path.join(out_dir, filename), (image * 255).astype(np.uint8))

        # Save quads
        self.log("results:", len(results_quad))
        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'quad.pkl'), 'wb') as f:
            pickle.dump(results_quad, f)

        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'quad_orig.pkl'), 'wb') as f:
            pickle.dump(results_orig_quad, f)

        # Save meta data
        results_new = []
        for img, P  in results_meta.items():
            img = os.path.basename(img)
            res = [format(r, '.6f') for r in P]
            results_new.append((img,res))
        with open(os.path.join(self.media_save_path["affine_transform"]["folder"], 'dataset.json'), 'w') as outfile:
            json.dump({"labels": results_new}, outfile, indent="\t")

    @torch.no_grad()
    def inject_ffhq_prior(self):
        """
        Run pretrained face restore model to inject ffhq prior
        """

        GFPGAN_LIB_PATH = os.path.join(os.getcwd(), 'submodules/GFPGAN')
        sys.path.append(GFPGAN_LIB_PATH)

        from tools.sr_utils import GFPGANer
        from basicsr.utils import imwrite

        self.log('++> Run GFPGAN for enhencing.')

        orig_images     = self.media_save_path["affine_transform"]["folder"]
        final_images    = self.media_save_path["inject_prior"]["folder"]

        os.makedirs(final_images, exist_ok=True)

        orig_img_list = sorted(glob.glob(os.path.join(orig_images, '*')))
        orig_img_list = [img for img in orig_img_list if not img.endswith('.json')]

        bg_upsampler        = None
        arch                = 'clean'
        channel_multiplier  = 2
        model_path          = self.weight_path["gfpgan"]

        restorer = GFPGANer(
            model_path          = model_path,
            upscale             = 2,
            arch                = arch,
            channel_multiplier  = channel_multiplier,
            bg_upsampler        = bg_upsampler
        )

        for img_path in orig_img_list:
            img_name = os.path.basename(img_path)
            self.log(f'Processing {img_name} ...')

            basename, ext   = os.path.splitext(img_name)
            input_img       = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # restore faces and background if necessary
            _, restored_faces, _ = restorer.enhance(
                input_img,
                has_aligned         = True,
                only_center_face    = False,
                paste_back          = False,
                weight              = 0.5)
            
            # save restored faces, which is a list (len == 1)
            if restored_faces is not None:
                extension = ext[1:]
                save_restore_path = os.path.join(final_images, f'{basename}.{extension}')
                imwrite(restored_faces.pop(), save_restore_path)

        self.log(f'Results are in the "{final_images}" folder.')

        # copy dataset.json
        shutil.copyfile(os.path.join(orig_images, 'dataset.json'), os.path.join(final_images, 'dataset.json'))
        self.enhenced   = True

        self.log('++> Enhencing finished.')

    def proceed_gan_inversion(self):
        """
        Run PTI
        """
        torch.cuda.empty_cache()

        if self.pretrained_type == 'spherehead':
            SPHEREHEAD_LIB_PAHT = os.path.join(os.getcwd(), 'submodules/SphereHead')
            sys.path.insert(1, SPHEREHEAD_LIB_PAHT)
            network_pkl = self.weight_path["3d-gan"]["spherehead"]
        elif self.pretrained_type == 'panohead':
            PANOHEAD_LIB_PATH = os.path.join(os.getcwd(), 'submodules/PanoHead')
            sys.path.insert(1, PANOHEAD_LIB_PATH)
            network_pkl = self.weight_path["3d-gan"]["panohead"]

        import dnnlib
        import legacy
        from training.dataset import ImageFolderDataset

        from tools.eg3d_utils.pti import project_multi_view, project_pti_multi_view, save_optimization_video

        outdir  = self.media_save_path["run_pti"]["folder"]
        os.makedirs(outdir, exist_ok=True)

        self.log('++> Proceed GAN inversion')
        self.log('++> Load Networks from "%s"...' % network_pkl)

        with dnnlib.util.open_url(network_pkl) as fp:
            network_data = legacy.load_network_pkl(fp)
            G = network_data['G_ema'].requires_grad_(True).to(self.device)

        # hard code
        G.rendering_kwargs["ray_start"] = 2.35

        if self.enhenced:
            dataset_path    = self.media_save_path["inject_prior"]["folder"]
        else:
            dataset_path    = self.media_save_path["affine_transform"]["folder"]

        dataset = ImageFolderDataset(
            path          = dataset_path,
            use_labels    = True,
            max_size      = None,
            xflip         = False
        )

        projected_w_steps   = project_multi_view(
            G,
            dataset,
            device      = self.device,
            log_fn      = self.log,
            num_steps   = self.pti_w_step,
        )

        G_steps = project_pti_multi_view(
            G,
            dataset,
            w_pivot     = projected_w_steps[-1:],
            device      = self.device,
            log_fn      = self.log,
            num_steps   = self.pti_finetune_step,
        )
        
        video = imageio.get_writer(
            os.path.join(self.media_save_path["video"]["folder"], 'optimization.mp4'),
            mode='I',
            fps=30,
            codec='libx264'
        )
        self.log(f'++> Saving optimization progress video ...')

        # save optimization video
        save_optimization_video(
            G,
            dataset,
            video,
            projected_w_steps,
            G_steps,
            device  = self.device
        )

        # Save final projected frame and W vector
        projected_w = projected_w_steps[-1]
        np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

        # Save network parameter
        with open(f'{outdir}/fintuned_generator.pkl', 'wb') as f:
            G_final = G_steps[-1].to(self.device)
            network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
            pickle.dump(network_data, f)

        self.log('++> GAN inversion finished')

    def render_inversion_result(self, orbit_frames=40, ele_list=[0]):
        """
        Render PTI result
        """
        torch.cuda.empty_cache()

        if self.pretrained_type == 'spherehead':
            SPHEREHEAD_LIB_PAHT = os.path.join(os.getcwd(), 'submodules/SphereHead')
            sys.path.insert(1, SPHEREHEAD_LIB_PAHT)
        elif self.pretrained_type == 'panohead':
            PANOHEAD_LIB_PATH = os.path.join(os.getcwd(), 'submodules/PanoHead')
            sys.path.insert(1, PANOHEAD_LIB_PATH)

        import dnnlib
        import legacy

        from tools.eg3d_utils.pti import gen_orbit_video

        outdir      = self.media_save_path["run_pti"]["folder"]

        self.log('++> Render PanoHead inversion')
        self.log(f'++> Load Networks from "{outdir}/fintuned_generator.pkl"...')

        ws = torch.tensor(np.load(f'{outdir}/projected_w.npz')['w']).to(self.device)
        with dnnlib.util.open_url(f'{outdir}/fintuned_generator.pkl') as fp:
            network_data = legacy.load_network_pkl(fp)
            G = network_data['G_ema'].requires_grad_(False).to(self.device) # type: ignore

        sampling_multiplier = 2
        G.rendering_kwargs["ray_start"]                     = 2.35 - self.rescale_factor / 2      # default: 2.35
        G.rendering_kwargs["ray_end"]                       = 3.3 + self.rescale_factor / 2      # default: 3.3
        G.rendering_kwargs['depth_resolution']              = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance']   = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

        gen_orbit_video(
            G,
            self.J,
            mp4_save_path       = os.path.join(self.media_save_path["video"]["folder"], 'pti.mp4'),
            save_path           = outdir,
            ws                  = ws,
            gs_lookat_point     = self.gs_camera_lookat_point,
            gs_radius           = self.gs_camera_radius,
            w_frames            = orbit_frames,
            ele_list            = ele_list,
            device              = self.device,
            rotate_type         = self.rotate_type,
            rescale_scene       = self.rescale_scene,
            rescale_factor      = self.rescale_factor,
        )

        self.log('++> Render finished')

    def execute_inverse_transform(self):
        """
        Paste aligned images into unaligned state
        """
        def natural_sort_key(s):
            return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
        
        self.log('++> Run affine inverse.')
        
        pti_dir         = self.media_save_path["run_pti"]["folder"]
        paste_dir       = self.media_save_path["inverse_transform"]["folder"]

        pti_image_dir   = os.path.join(pti_dir, 'image')
        # pano_mask_dir   = os.path.join(pano_dir, 'mask')

        images_files    = os.listdir(pti_image_dir)
        # mask_files      = os.listdir(pano_mask_dir)

        # for proper visualzation
        images_files = sorted(images_files, key=natural_sort_key)

        save_image_dir  = os.path.join(paste_dir, 'image')
        # save_mask_dir   = os.path.join(paste_dir, 'mask')
        save_video_dir  = os.path.join(paste_dir, 'paste_novel_view.mp4')

        os.makedirs(save_image_dir, exist_ok=True)
        # os.makedirs(save_mask_dir, exist_ok=True)

        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'quad.pkl'), "rb") as f:
            inputs = pickle.load(f, encoding="latin1")

        quad = next((value for key, value in inputs.items() if os.path.basename(key) == '0001.png'), None)
        if quad is None:
            raise ValueError
        
        size = 512

        # magic numbers
        top_expand      = 0.1
        left_expand     = 0.05
        bottom_expand   = 0.0
        right_expand    = 0.05

        crop_w      = int(size * (1 + left_expand + right_expand))
        crop_h      = int(size * (1 + top_expand + bottom_expand))
        crop_size   = (crop_w, crop_h)

        top     = int(size * top_expand)
        left    = int(size * left_expand)

        bound = np.array([
                        [left, top],
                        [left, top + size - 1],
                        [left + size - 1, top + size - 1],
                        [left + size - 1, top]
                        ], dtype=np.float32)

        delta_bound = bound - 256
        if self.rescale_scene:
            ratio = 2.7 / (2.7 + self.rescale_factor)
            delta_bound *= ratio
        else:
            delta_bound *= 1.0
        bound = delta_bound + 256
        
        dst_points = quad

        # prepare inverse transformation
        M, _ = cv2.estimateAffine2D(dst_points, bound)
        M = M.flatten()

        all_images_np = []

        for i in range(len(images_files)):

            rgb_image   = PIL.Image.new("RGB", (size, size), "white")
            # mask_image  = PIL.Image.new("L", (size, size), "black")

            image_file  = images_files[i]
            # mask_file   = mask_files[i]

            image   = PIL.Image.open(os.path.join(pti_image_dir, image_file)).convert('RGB')
            # mask    = PIL.Image.open(os.path.join(pano_mask_dir, mask_file)).convert('L')

            # mask = mask.point(lambda x: 255 if x > 128 else 0, mode='1')

            image   = image.resize(crop_size)
            # mask    = mask.resize(crop_size)
        
            unalign_img = image.transform(crop_size, PIL.Image.Transform.AFFINE, M, PIL.Image.Resampling.BICUBIC)
            # unalign_mask = mask.transform(crop_size, PIL.Image.Transform.AFFINE, M, PIL.Image.Resampling.BICUBIC)

            unalign_img_np = np.array(unalign_img)
            mask_ops = np.all(unalign_img_np == [0, 0, 0], axis=-1)
            mask_ops = PIL.Image.fromarray(mask_ops.astype(np.uint8) * 255).convert("L")

            rgb_image.paste(unalign_img, (0, 0),    mask=PIL.ImageOps.invert(mask_ops))
            # mask_image.paste(unalign_mask, (0, 0),  mask=PIL.ImageOps.invert(mask_ops))

            rgb_image.save(os.path.join(save_image_dir, image_file))
            # mask_image.save(os.path.join(save_mask_dir, mask_file))

            all_images_np.append(np.array(rgb_image))

        imageio.mimwrite(save_video_dir, all_images_np, fps=25, quality=8, macro_block_size=1)
        shutil.copyfile(os.path.join(pti_dir, 'trajectory.json'), os.path.join(paste_dir, 'trajectory.json'))

        self.log('++> Affine inverse finished.')

    @torch.no_grad()
    def retrieve_image_mask(self, input_path = None, output_path = None):
        """
        Get mask from parsing net
        """
        PARSING_LIB_PATH = os.path.join(os.getcwd(), 'submodules/face-parsing.PyTorch')
        sys.path.append(PARSING_LIB_PATH)

        import importlib.util
        spec = importlib.util.spec_from_file_location("BiSeNetModule", os.path.join(PARSING_LIB_PATH, 'model.py'))
        bi_se_net_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bi_se_net_module)

        BiSeNet = bi_se_net_module.BiSeNet

        self.log('++> Matting pasted images.')
                
        paste_dir           = self.media_save_path["inverse_transform"]["folder"]
        paste_image_dir     = os.path.join(paste_dir, 'image')
        paste_mask_dir      = os.path.join(paste_dir, 'mask_bisenet')
        os.makedirs(paste_mask_dir, exist_ok=True)

        if input_path is None or output_path is None:
            input_path  = paste_image_dir
            output_path = paste_mask_dir
        else:
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)

        n_classes = 19
        net = BiSeNet(n_classes=n_classes).to(self.device)
        ckpt_path = self.weight_path["bisenet"]
        net.load_state_dict(torch.load(ckpt_path))
        net.eval()

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

        im_names = os.listdir(input_path)
        im_names = [img for img in im_names if not img.endswith('.json')]
        for im_name in im_names:
            self.log('Matte image: {0}'.format(im_name))

            image = PIL.Image.open(os.path.join(input_path, im_name))
            info = {}

            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)

            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            single_label_mask = np.zeros_like(parsing)
            head_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])
            # head_array = np.array([16])
            index = np.where(np.isin(parsing, head_array))     # neckhead
            single_label_mask[index] = 1
            single_label_mask_image = PIL.Image.fromarray((single_label_mask).astype(np.uint8) * 255)

            matte_name = im_name
            single_label_mask_image.save(os.path.join(output_path, matte_name))

            # image_w_mask = PIL.Image.composite(image, PIL.Image.new('RGB', image.size, (255, 255, 255)), single_label_mask_image)
            # image_w_mask.save(os.path.join(input_path, matte_name))

        self.log('++> Matted finished.')

    @torch.no_grad()
    def retrieve_image_mask_modnet(self, input_path = None, output_path = None):
        """
        Get mask from MODNet
        """

        MODNET_LIB_PATH = os.path.join(os.getcwd(), 'submodules/MODNet')
        sys.path.append(MODNET_LIB_PATH)

        from src.models.modnet import MODNet

        self.log('++> Matting pasted images.')
                
        paste_dir           = self.media_save_path["inverse_transform"]["folder"]
        paste_image_dir     = os.path.join(paste_dir, 'image')
        paste_mask_dir      = os.path.join(paste_dir, 'mask_modnet')
        os.makedirs(paste_mask_dir, exist_ok=True)

        if input_path is None or output_path is None:
            input_path  = paste_image_dir
            output_path = paste_mask_dir
        else:
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)

        ckpt_path           = './weights/modnet_webcam_portrait_matting.ckpt'

        # define hyper-parameters
        ref_size = 512

        # define image to tensor transform
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # create MODNet and load the pre-trained ckpt
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)

        modnet = modnet.to(self.device)
        weights = torch.load(ckpt_path)
        modnet.load_state_dict(weights)
        modnet.eval()

        # inference images
        im_names = os.listdir(input_path)
        im_names = [img for img in im_names if not img.endswith('.json')]
        for im_name in im_names:
            self.log('Matte image: {0}'.format(im_name))

            # read image
            im = PIL.Image.open(os.path.join(input_path, im_name))

            # unify image channels to 3
            im = np.asarray(im)
            if len(im.shape) == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            elif im.shape[2] == 4:
                im = im[:, :, 0:3]

            # convert image to PyTorch tensor
            im = PIL.Image.fromarray(im)
            im = im_transform(im)

            # add mini-batch dim
            im = im[None, :, :, :]

            # resize image for input
            im_b, im_c, im_h, im_w = im.shape
            if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                if im_w >= im_h:
                    im_rh = ref_size
                    im_rw = int(im_w / im_h * ref_size)
                elif im_w < im_h:
                    im_rw = ref_size
                    im_rh = int(im_h / im_w * ref_size)
            else:
                im_rh = im_h
                im_rw = im_w
            
            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32
            im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

            # inference
            _, _, matte = modnet(im.to(self.device), True)

            # resize and save matte
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
            matte_name = im_name
            PIL.Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(output_path, matte_name))


        mask_files = [f for f in os.listdir(paste_mask_dir) if f.endswith('.png')]
        result = None

        for mask_file in mask_files:
            img = PIL.Image.open(os.path.join(paste_mask_dir, mask_file))
            img = np.array(img)

            if result is None:
                result = img
            else:
                result = np.logical_and(result, img)

        result_img = PIL.Image.fromarray(result.astype(np.uint8) * 255)

        threshold = 10

        boundary_height = None
        for row in range(result.shape[1]):
            row_ = 512 - row - 1
            white_count = np.sum(result[row_, :])
            if white_count >= threshold:
                boundary_height = row_
                break

        if boundary_height is not None:
            boundary_height = row_
        else:
            raise

        width = ref_size
        height = ref_size

        new_img = np.zeros((height, width), dtype=np.uint8)

        new_img[boundary_height:, :] = 0
        new_img[:boundary_height, :] = 255

        new_img_pil = PIL.Image.fromarray(new_img)

        new_img_pil.save(os.path.join(self.media_save_path["inverse_transform"]["folder"], 'torsor_boundary.png'))

        self.log('++> Matted finished.')

    def heatmap_check(self):
        """
        Check misalignment via heatmap
        """

        from tools.util import colorize_weights_map

        self.log('++> Run heatmap check.')

        paste_dir           = self.media_save_path["inverse_transform"]["folder"]
        paste_image_dir     = os.path.join(paste_dir, 'image')
        gs_image_dir        = self.media_save_path["render_novel_view"]["folder"]

        heatmap_dir         = self.media_save_path["heatmap_check"]["folder"]
        os.makedirs(heatmap_dir, exist_ok=True)

        img_names = os.listdir(gs_image_dir)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        for img_name in img_names:
            
            gs_img      = PIL.Image.open(os.path.join(gs_image_dir,     img_name))
            paste_img   = PIL.Image.open(os.path.join(paste_image_dir,  img_name))

            gs_img_tensor       = transform(gs_img)[None, ...]
            paste_img_tensor    = transform(paste_img)[None, ...]

            err = (gs_img_tensor - paste_img_tensor).abs().max(dim=1)[0].clip(0, 1)
            err_plot = colorize_weights_map(err, min_val=0, max_val=1)

            grid   = torchvision.utils.make_grid(err_plot, nrow=1, normalize=True, value_range=(0, 1))
            torchvision.utils.save_image(grid, os.path.join(heatmap_dir, img_name))

        self.log('++> Heatmap check done.')



        




                    