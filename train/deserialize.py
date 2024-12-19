import os
import glob
import torch

#-------------------------------------------------------------------------------#

def deserialize_checkpoints_fateavatar(self, checkpoint_dict):

    # --- separate gaussian attributes, since 'load_state_dict' can not handle it directly --- #
    gaussian_attributes = ['_offset', '_features_dc', '_features_rest', 
                            '_scaling','_rotation', '_opacity',
                            'face_index', 'bary_coords']
    gaussian_dict = {key: checkpoint_dict['model'].pop(key) for key in gaussian_attributes}

    # --- load other attributes (if have) --- #
    missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)

    assert missing_keys == gaussian_attributes

    # --- load gaussian attributes --- #
    for attr_name, attr_value in gaussian_dict.items():
        if attr_name not in ['face_index', 'bary_coords']:
            setattr(self.model, attr_name, torch.nn.Parameter(attr_value.requires_grad_(True)))
        else:
            setattr(self.model, attr_name, attr_value)
        missing_keys.remove(attr_name)

    # --- overwrite number of points --- #
    self.model.num_points = gaussian_dict['_offset'].shape[0]

    self.model.max_radii2D        = torch.zeros((self.model.num_points), device=self.device)
    self.model.xyz_gradient_accum = torch.zeros((self.model.num_points, 1), device=self.device)
    self.model.denom              = torch.zeros((self.model.num_points, 1), device=self.device)
    self.model.sample_flag        = torch.zeros((self.model.num_points), device=self.device)

    self.log("[INFO] loaded model.")
    if len(missing_keys) > 0:
        self.log(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        self.log(f"[WARN] unexpected keys: {unexpected_keys}")

#-------------------------------------------------------------------------------#

def deserialize_checkpoints_flashavatar(self, checkpoint_dict):

    # --- load other attributes (if have) --- #
    missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)

    self.log("[INFO] loaded model.")
    if len(missing_keys) > 0:
        self.log(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        self.log(f"[WARN] unexpected keys: {unexpected_keys}")

#-------------------------------------------------------------------------------#

def deserialize_checkpoints_gaussianavatars(self, checkpoint_dict):

    # --- separate gaussian attributes, since 'load_state_dict' can not handle it directly --- #
    gaussian_attributes = ['_xyz', '_features_dc', '_features_rest', 
                            '_scaling','_rotation', '_opacity',
                            'binding', 'binding_counter']
    gaussian_dict = {key: checkpoint_dict['model'].pop(key) for key in gaussian_attributes}

    # --- load other attributes (if have) --- #
    missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)

    assert missing_keys == gaussian_attributes

    # --- load gaussian attributes --- #
    for attr_name, attr_value in gaussian_dict.items():
        if attr_name not in ['binding', 'binding_counter']:
            setattr(self.model, attr_name, torch.nn.Parameter(attr_value.requires_grad_(True)))
        else:
            setattr(self.model, attr_name, attr_value)
        missing_keys.remove(attr_name)

    # --- overwrite number of points --- #
    self.model.num_points = gaussian_dict['_xyz'].shape[0]

    self.model.max_radii2D        = torch.zeros((self.model.num_points), device=self.device)
    self.model.xyz_gradient_accum = torch.zeros((self.model.num_points, 1), device=self.device)
    self.model.denom              = torch.zeros((self.model.num_points, 1), device=self.device)

    self.log("[INFO] loaded model.")
    if len(missing_keys) > 0:
        self.log(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        self.log(f"[WARN] unexpected keys: {unexpected_keys}")

#-------------------------------------------------------------------------------#

def deserialize_checkpoints_monogaussianavatar(self, checkpoint_dict):

    pc_attributes = ['pc.points']
    pc_dict = {key: checkpoint_dict['model'].pop(key) for key in pc_attributes}

    # --- load other attributes (if have) --- #
    missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)

    assert missing_keys == pc_attributes

    self.model.pc.points.data = pc_dict['pc.points']
    # --- overwrite number of points --- #
    self.model.num_points = pc_dict['pc.points'].shape[0]
    self.model.radius = 0.0007610908653441582   # i just ... slack off
    self.model.visible_points = torch.zeros(self.model.pc.points.shape[0]).bool().to(self.model.device)

    missing_keys.remove('pc.points')

    self.log("[INFO] loaded model.")
    if len(missing_keys) > 0:
        self.log(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        self.log(f"[WARN] unexpected keys: {unexpected_keys}")

#-------------------------------------------------------------------------------#

def deserialize_checkpoints_splattingavatar(self, checkpoint_dict):

    gaussian_attributes = ['_uvd', '_scaling', '_rotation', 
                            '_opacity','_features_dc', '_features_rest', 'sample_fidxs', 'sample_bary']
    gaussian_dict = {key: checkpoint_dict['model'].pop(key) for key in gaussian_attributes}

    # --- load other attributes (if have) --- #
    missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)

    # --- load gaussian attributes --- #
    for attr_name, attr_value in gaussian_dict.items():
        if attr_name not in ['sample_fidxs', 'sample_bary']:
            setattr(self.model, attr_name, torch.nn.Parameter(attr_value.requires_grad_(True)))
        else:
            setattr(self.model, attr_name, attr_value)
        missing_keys.remove(attr_name)

    # --- overwrite number of points --- #
    self.model.num_points = gaussian_dict['_uvd'].shape[0]

    self.model.max_radii2D        = torch.zeros((self.model.num_points), device=self.device)
    self.model.xyz_gradient_accum = torch.zeros((self.model.num_points, 1), device=self.device)
    self.model.denom              = torch.zeros((self.model.num_points, 1), device=self.device)

    self.log("[INFO] loaded model.")
    if len(missing_keys) > 0:
        self.log(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        self.log(f"[WARN] unexpected keys: {unexpected_keys}")

#-------------------------------------------------------------------------------#