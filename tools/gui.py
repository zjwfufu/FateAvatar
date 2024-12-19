import  os
import  time
import  torch
import  numpy as np
import  dearpygui.dearpygui as dpg

from    volume_rendering.camera_3dgs   import Camera
from    pytorch3d.transforms           import matrix_to_axis_angle
from    scipy.spatial.transform import Rotation as R


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1., is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


class ViewerMouseCamera:
    def __init__(self, r, W=800, H=800, fovx=0.33639895755220545, fovy=0.33254485845058807,
                 near=0.01, far=100.0):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovx = fovx
        self.fovy = fovy
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])


class Viewer:
    def __init__(self, opt, cfg, avatar, loader, identity_dict):
        self.opt = opt
        self.cfg = cfg
        self.cfg_data = self.cfg.dataset
        self.cfg_model = self.cfg.model
        self.cfg_loss = self.cfg.loss
        self.device = self.opt.device

        self.model = avatar
        self.loader = loader

        # displayed image size
        self.W = 800
        self.H = 800

        # GUI modes: manual, autoplay_on_train, autoplay_on_test
        self.current_mode = 'manual'
        # for autoplay mode
        self.train_idx = 0
        self.test_idx = 0

        # training expression and pose
        self.train_expr = self.loader.train_expression
        self.train_pose = self.loader.train_flame_pose
        self.train_num = self.train_pose.shape[0]

        # testing expression and pose
        self.test_expr = self.loader.test_expression
        self.test_pose = self.loader.test_flame_pose
        self.test_num = self.test_pose.shape[0]

        # current expression and pose
        self.expr = self.model.flame.canonical_exp
        self.pose = self.model.flame.canonical_pose

        # backup canonical expression and pose
        self.expr_bkp = self.expr.clone()
        self.pose_bkp = self.pose.clone()

        self.radius = self.loader.train_cam_pose[0, :].detach().cpu().numpy()[2]
        self.mouse_cam = ViewerMouseCamera(r=self.radius)

        self.R = torch.tensor([[[1.0000, 0.0000, 0.0000],
                                [0.0000, -1.0000, 0.0000],
                                [0.0000, 0.0000, -1.0000]]])
        self.T = self.loader.train_cam_pose[0:1, :].detach().cpu()

        self.init_fovx = identity_dict['camera_fovx'] * 180 / np.pi
        self.init_fovy = identity_dict['camera_fovy'] * 180 / np.pi

        self.fovx = identity_dict['camera_fovx'] * 180 / np.pi
        self.fovy = identity_dict['camera_fovy'] * 180 / np.pi
        self.orbit_cam = {'use': False, 'azimuth': 0, 'elevation': 0, 'radius': self.radius}

        # buffer image for displaying
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer image

        dpg.create_context()
        self.register_dpg()
        self.test_step()

    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
                tag="_primary_window",
                width=self.W,
                height=self.H,
                pos=[0, 0],
                no_move=True,
                no_title_bar=True,
                no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
                label="Control",
                tag="_control_window",
                width=600,
                height=self.H,
                pos=[self.W, 0],
                no_move=True,
                no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # rendering options
            with dpg.collapsing_header(label="CAMERA", default_open=True):
                # fov slider
                def callback_set_fovx(sender, app_data):
                    self.fovx = app_data
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoVX (horizontal)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=self.fovx,
                    callback=callback_set_fovx,
                    tag='_slider_fovx',
                )

                def callback_set_fovy(sender, app_data):
                    self.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoVY (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=self.fovy,
                    callback=callback_set_fovy,
                    tag='_slider_fovy',
                )

                def callback_set_azimuth(sender, app_data):
                    self.orbit_cam['azimuth'] = app_data
                    self.orbit_cam['use'] = True
                    self.need_update = True

                dpg.add_slider_int(
                    label="azimuth",
                    min_value=-179,
                    max_value=180,
                    format="%d deg",
                    default_value=self.orbit_cam['azimuth'],
                    callback=callback_set_azimuth,
                    tag='_slider_azimuth',
                )

                def callback_set_elevation(sender, app_data):
                    self.orbit_cam['elevation'] = app_data
                    self.orbit_cam['use'] = True
                    self.need_update = True

                dpg.add_slider_int(
                    label="elevation",
                    min_value=-89,
                    max_value=89,
                    format="%d deg",
                    default_value=self.orbit_cam['elevation'],
                    callback=callback_set_elevation,
                    tag='_slider_elevation',
                )

                def callback_set_radius(sender, app_data):
                    self.orbit_cam['radius'] = app_data
                    self.orbit_cam['use'] = True
                    self.need_update = True

                dpg.add_slider_float(
                    label="radius",
                    min_value=1.0,
                    max_value=10.0,
                    format="%.2f",
                    default_value=self.orbit_cam['radius'],
                    callback=callback_set_radius,
                    tag='_slider_radius',
                )

                def callback_reset_cam(sender, app_data):
                    self.orbit_cam['radius'] = self.radius
                    self.orbit_cam['azimuth'] = 0
                    self.orbit_cam['elevation'] = 0
                    self.fovx = self.init_fovx
                    self.fovy = self.init_fovy
                    self.mouse_cam = ViewerMouseCamera(r=self.radius)  # reset mouse cam
                    self.orbit_cam['use'] = True

                    dpg.set_value("_slider_azimuth", 0)
                    dpg.set_value("_slider_elevation", 0)
                    dpg.set_value("_slider_radius", self.radius)
                    dpg.set_value("_slider_fovx", 19)
                    dpg.set_value("_slider_fovy", 19)

                    self.need_update = True

                dpg.add_button(
                    label="Reset camera",
                    tag="_button_reset_cam",
                    callback=callback_reset_cam,
                )

            # FLAME parameters
            with dpg.collapsing_header(label="FLAME parameters", default_open=True):
                joint_id = {'root': 0, 'neck': 3, 'jaw': 6, 'left_eyes': 9, 'right_eyes': 12}
                axis_id = {'x': 0, 'y': 1, 'z': 2}

                def callback_set_pose(sender, app_data):
                    joint, axis = sender.split('-')[1:3]
                    axis_idx = axis_id[axis]
                    joint_idx = joint_id[joint]
                    self.pose[0, joint_idx + axis_idx] = app_data
                    self.need_update = True

                dpg.add_text(f'Joints')
                self.pose_sliders = []
                for joint in ['root', 'neck', 'jaw', 'left_eyes', 'right_eyes']:
                    max_rot = 0.10 if 'eyes' in joint else 0.30
                    with dpg.group(horizontal=True):
                        dpg.add_slider_float(min_value=-max_rot, max_value=max_rot,
                                             format="%.2f",
                                             default_value=self.pose[0, joint_id[joint] + 0],
                                             callback=callback_set_pose,
                                             tag=f"_slider-{joint}-x", width=150)
                        dpg.add_slider_float(min_value=-max_rot, max_value=max_rot,
                                             format="%.2f",
                                             default_value=self.pose[0, joint_id[joint] + 1],
                                             callback=callback_set_pose,
                                             tag=f"_slider-{joint}-y", width=150)
                        dpg.add_slider_float(min_value=-max_rot, max_value=max_rot,
                                             format="%.2f",
                                             default_value=self.pose[0, joint_id[joint] + 2],
                                             callback=callback_set_pose,
                                             tag=f"_slider-{joint}-z", width=150)
                        self.pose_sliders.append(f"_slider-{joint}-x")
                        self.pose_sliders.append(f"_slider-{joint}-y")
                        self.pose_sliders.append(f"_slider-{joint}-z")
                        dpg.add_text(f'{joint:4s}')
                dpg.add_text('         roll                  pitch                  yaw')

                dpg.add_separator()

                def callback_set_expr(sender, app_data):
                    expr_i = int(sender.split('-')[2])
                    self.expr[0, expr_i] = app_data
                    self.need_update = True

                self.expr_sliders = []
                dpg.add_text(f'Expressions')
                # Randomly select 10 expressions out of 50
                # expr_rand = sorted(random.sample(range(0, 49), 10))
                expr_rand = range(0, 10)
                max_expr = 1.5
                for i in expr_rand:
                    dpg.add_slider_float(label=f"{i}", min_value=-max_expr, max_value=max_expr,
                                         format="%.4f", default_value=self.expr[0, i],
                                         callback=callback_set_expr, tag=f"_slider-expr-{i}",
                                         width=500)
                    self.expr_sliders.append(f"_slider-expr-{i}")

                def callback_reset_flame(sender, app_data):
                    self.expr = self.expr_bkp.clone()
                    self.pose = self.pose_bkp.clone()
                    self.need_update = True

                    for idx, pose_slider in enumerate(self.pose_sliders):
                        dpg.set_value(pose_slider, self.pose_bkp[0, idx])
                    for idx, expr_slider in enumerate(self.expr_sliders):
                        idx_rand = int(expr_slider.split('-')[2])
                        dpg.set_value(expr_slider, self.expr_bkp[0, idx_rand])

                dpg.add_button(label="Reset FLAME", tag="_button_reset_flame",
                               callback=callback_reset_flame)

            # play on train dataset or test dataset
            with dpg.collapsing_header(label="PLAYING", default_open=True):

                def callback_play_train(sender, app_data):
                    self.pose = self.train_pose[app_data:app_data + 1].clone()
                    self.expr = self.train_expr[app_data:app_data + 1].clone()

                    for idx, pose_slider in enumerate(self.pose_sliders):
                        dpg.set_value(pose_slider, self.pose[0, idx])
                    for idx, expr_slider in enumerate(self.expr_sliders):
                        idx_rand = int(expr_slider.split('-')[2])
                        dpg.set_value(expr_slider, self.expr[0, idx_rand])

                    self.train_idx = app_data

                    self.need_update = True

                dpg.add_slider_int(
                    label="train",
                    min_value=0,
                    max_value=self.train_num - 1,
                    format="%d",
                    default_value=0,
                    callback=callback_play_train,
                    tag='_slider_play_train',
                    width=500
                )

                def callback_play_test(sender, app_data):
                    self.pose = self.test_pose[app_data:app_data + 1].clone()
                    self.expr = self.test_expr[app_data:app_data + 1].clone()

                    for idx, pose_slider in enumerate(self.pose_sliders):
                        dpg.set_value(pose_slider, self.pose[0, idx])
                    for idx, expr_slider in enumerate(self.expr_sliders):
                        idx_rand = int(expr_slider.split('-')[2])
                        dpg.set_value(expr_slider, self.expr[0, idx_rand])

                    self.test_idx = app_data

                    self.need_update = True

                dpg.add_slider_int(
                    label="test",
                    min_value=0,
                    max_value=self.test_num - 1,
                    format="%d",
                    default_value=0,
                    callback=callback_play_test,
                    tag='_slider_play_test',
                    width=500
                )

                dpg.add_separator()

                def callback_autoplay_train(sender, app_data):
                    self.current_mode = 'autoplay_on_train'
                    self.need_update = True

                def callback_autoplay_test(sender, app_data):
                    self.current_mode = 'autoplay_on_test'
                    self.need_update = True

                def callback_manual(sender, app_data):
                    self.current_mode = 'manual'
                    self.need_update = True

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Autoplay on train", tag="_button_autoplay_train",
                                   callback=callback_autoplay_train)

                    dpg.add_button(label="Autoplay on test", tag="_button_autoplay_test",
                                   callback=callback_autoplay_test)

                    dpg.add_button(label="Stop", tag="_button_manual", callback=callback_manual)

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.mouse_cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.mouse_cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.mouse_cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="FATE",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    # for autoplay on train or test dataset
    def play_step(self):
        if self.current_mode == 'autoplay_on_train':
            self.pose = self.train_pose[self.train_idx:self.train_idx + 1].clone()
            self.expr = self.train_expr[self.train_idx:self.train_idx + 1].clone()

            for idx, pose_slider in enumerate(self.pose_sliders):
                dpg.set_value(pose_slider, self.pose[0, idx])
            for idx, expr_slider in enumerate(self.expr_sliders):
                idx_rand = int(expr_slider.split('-')[2])
                dpg.set_value(expr_slider, self.expr[0, idx_rand])

            dpg.set_value('_slider_play_train', self.train_idx)
            self.train_idx += 1
            if self.train_idx >= self.train_num:
                self.train_idx = 0

        elif self.current_mode == 'autoplay_on_test':
            self.pose = self.test_pose[self.test_idx:self.test_idx + 1].clone()
            self.expr = self.test_expr[self.test_idx:self.test_idx + 1].clone()

            for idx, pose_slider in enumerate(self.pose_sliders):
                dpg.set_value(pose_slider, self.pose[0, idx])
            for idx, expr_slider in enumerate(self.expr_sliders):
                idx_rand = int(expr_slider.split('-')[2])
                dpg.set_value(expr_slider, self.expr[0, idx_rand])

            dpg.set_value('_slider_play_test', self.test_idx)
            self.test_idx += 1
            if self.test_idx >= self.test_num:
                self.test_idx = 0

        self.need_update = True

    def test_step(self):
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # if self.need_update:
        if self.orbit_cam['use']:
            pose = orbit_camera(self.orbit_cam['elevation'], self.orbit_cam['azimuth'],
                                self.orbit_cam['radius'])
            self.mouse_cam.rot = R.from_matrix(pose[:3, :3])
            self.mouse_cam.radius = np.linalg.norm(pose[:3, 3])

        pose = self.mouse_cam.pose
        r = np.linalg.norm(pose[:3, 3])
        self.T[0, 2] = torch.tensor(r)

        rot = matrix_to_axis_angle(
            torch.tensor(pose[:3, :3], device=self.device, dtype=torch.float32))
        self.pose_render = self.pose.clone()
        self.pose_render[0, :3] -= rot

        if not self.orbit_cam['use']:
            rot_np = rot.cpu().numpy()
            self.orbit_cam['elevation'] = int(np.rad2deg(rot_np[0]))
            self.orbit_cam['azimuth'] = int(np.rad2deg(rot_np[1]))
            self.orbit_cam['radius'] = r

            dpg.set_value("_slider_azimuth", int(np.rad2deg(rot_np[1])))
            dpg.set_value("_slider_elevation", int(np.rad2deg(rot_np[0])))
            dpg.set_value("_slider_radius", r)

        cur_cam = Camera(R=self.R, T=self.T, FoVx=np.deg2rad(self.fovx),
                         FoVy=np.deg2rad(self.fovy), img_res=(self.W, self.H))

        out = self.model.inference(
            expression  = self.expr,
            flame_pose  = self.pose_render,
            camera      = cur_cam
        )

        self.buffer_image = (
            out.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )  # buffer must be contiguous, else seg fault!

        self.orbit_cam['use'] = False
        self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000 / t)} FPS)")
        dpg.set_value("_texture",
                      self.buffer_image)  # buffer must be contiguous, else seg fault!

    def render(self):
        while dpg.is_dearpygui_running():
            if self.current_mode != 'manual':
                self.play_step()
                time.sleep(0.02)    # for reasonable play rate
            self.test_step()
            dpg.render_dearpygui_frame()