import math
import torch
import torch.nn as nn

from pytorch3d.structures       import Meshes
from pytorch3d.io               import load_obj
from pytorch3d.renderer.mesh    import rasterize_meshes
from pytorch3d.ops              import mesh_face_areas_normals

#-------------------------------------------------------------------------------#

# modified from https://github.com/facebookresearch/pytorch3d
class Pytorch3dRasterizer(nn.Module):
    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
            'cull_backfaces': True
        }
        # raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings['image_size']
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings['blur_radius'],
            faces_per_pixel=raster_settings['faces_per_pixel'],
            bin_size=raster_settings['bin_size'],
            max_faces_per_bin=raster_settings['max_faces_per_bin'],
            perspective_correct=raster_settings['perspective_correct'],
            cull_backfaces=raster_settings['cull_backfaces']
        )

        return pix_to_face, bary_coords
    
#-------------------------------------------------------------------------------#

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    Indexing the coordinates of the three vertices on each face.

    Args:
        vertices:   [bs, V, 3]
        faces:      [bs, F, 3]

    Return: 
        face_to_vertices: [bs, F, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    # assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def uniform_sampling_barycoords(
        num_points:  int,
        tex_coord:   torch.Tensor,
        uv_faces:    torch.Tensor,
        d_size:      float=1.0,
        strict:      bool=True
    ):
    """
    Uniformly sampling barycentric coordinates using the rasterizer.

    Args:
        num_points:     int                 sampling points number
        tex_coord:      [5150, 2]           UV coords for each vert
        uv_faces:       [F,3]               UV faces to UV coords index
        d_size:         const               to control sampling points number
    Returns:
        face_index      [num_points]        save which face each bary_coords belongs to
        bary_coords     [num_points, 3]
    """
    
    uv_size = int(math.sqrt(num_points) * d_size)
    uv_rasterizer = Pytorch3dRasterizer(uv_size)

    tex_coord   = tex_coord[None, ...]
    uv_faces    = uv_faces[None, ...]

    tex_coord_ = torch.cat([tex_coord, tex_coord[:,:,0:1]*0.+1.], -1)
    tex_coord_ = tex_coord_ * 2 - 1 
    tex_coord_[...,1] = - tex_coord_[...,1]

    pix_to_face, bary_coords = uv_rasterizer(tex_coord_.expand(1, -1, -1), uv_faces.expand(1, -1, -1))
    mask = (pix_to_face == -1)

    face_index = pix_to_face[~mask]
    bary_coords = bary_coords[~mask]

    cur_n = face_index.shape[0]

    # fix sampling number to num_points
    if strict:
        if cur_n < num_points:
            pad_size        = num_points - cur_n
            new_face_index  = face_index[torch.randint(0, cur_n, (pad_size,))]
            new_bary_coords = torch.rand((pad_size, 3), device=bary_coords.device)
            new_bary_coords = new_bary_coords / new_bary_coords.sum(dim=-1, keepdim=True)
            face_index      = torch.cat([face_index, new_face_index], dim=0)
            bary_coords     = torch.cat([bary_coords, new_bary_coords], dim=0)
        elif cur_n > num_points:
            face_index  = face_index[:num_points]
            bary_coords = bary_coords[:num_points]


    return face_index, bary_coords

def random_sampling_barycoords(
        num_points:   int,
        vertices:     torch.Tensor,
        faces:        torch.Tensor
    ):
    """
    Randomly sampling barycentric coordinates using the rasterizer.

    Args:
        num_points:     int                 sampling points number
        vertices:       [V, 3]           
        faces:          [F,3]
    Returns:
        face_index      [num_points]        save which face each bary_coords belongs to
        bary_coords     [num_points, 3]
    """

    areas, _ = mesh_face_areas_normals(vertices.squeeze(0), faces)

    g1 = torch.Generator(device=vertices.device)
    g1.manual_seed(0)

    face_index = areas.multinomial(
            num_points, replacement=True, generator=g1
        )  # (N, num_samples)

    uvw = torch.rand((face_index.shape[0], 3), device=vertices.device)
    bary_coords = uvw / uvw.sum(dim=-1, keepdim=True)

    return face_index, bary_coords

def reweight_verts_by_barycoords(
        verts:       torch.Tensor,
        faces:       torch.Tensor,
        face_index:  torch.Tensor,
        bary_coords: torch.Tensor,
    ):
    """
    Reweights the vertices based on the barycentric coordinates for each face.

    Args:
        verts:          [bs, V, 3].
        faces:          [F, 3]
        face_index:     [N].
        bary_coords:    [N, 3].

    Returns:
        Reweighted vertex positions of shape [bs, N, 3].
    """
    
    # index attributes by face
    B               = verts.shape[0]
    face_verts      = face_vertices(verts,  faces.expand(B, -1, -1))   # [1, F, 3, 3]
    # gather idnex for every splat
    N               = face_index.shape[0]
    face_index_3    = face_index.view(1, N, 1, 1).expand(B, N, 3, 3)
    position_vals   = face_verts.gather(1, face_index_3)
    # reweight
    position_vals   = (bary_coords[..., None] * position_vals).sum(dim = -2)

    return position_vals

def reweight_uvcoords_by_barycoords(
        uvcoords:    torch.Tensor,
        uvfaces:     torch.Tensor,
        face_index:  torch.Tensor,
        bary_coords: torch.Tensor,
    ):
    """
    Reweights the UV coordinates based on the barycentric coordinates for each face.

    Args:
        uvcoords:       [bs, V', 2].
        uvfaces:        [F, 3].
        face_index:     [N].
        bary_coords:    [N, 3].

    Returns:
        Reweighted UV coordinates, shape [bs, N, 2].
    """

    # homogeneous coordinates
    num_v           = uvcoords.shape[0]
    uvcoords        = torch.cat([uvcoords, torch.ones((num_v, 1)).to(uvcoords.device)], dim=1)
    # index attributes by face
    uvcoords        = uvcoords[None, ...]
    face_verts      = face_vertices(uvcoords,  uvfaces.expand(1, -1, -1))   # [1, F, 3, 3]
    # gather idnex for every splat
    N               = face_index.shape[0]
    face_index_3    = face_index.view(1, N, 1, 1).expand(1, N, 3, 3)
    position_vals   = face_verts.gather(1, face_index_3)
    # reweight
    position_vals   = (bary_coords[..., None] * position_vals).sum(dim = -2)

    return position_vals

# modified from https://github.com/computational-imaging/GSM/blob/main/main/gsm/deformer/util.py
def get_shell_verts_from_base(
        template_verts: torch.Tensor,
        template_faces: torch.Tensor,
        offset_len: float,
        num_shells: int,
        deflat = False,
    ):
    """
    Generates shell vertices by offsetting the original mesh's vertices along their normals.

    Args:
        template_verts: [bs, V, 3].
        template_faces: [F, 3].
        offset_len:     Positive number specifying the offset length for generating shells.
        num_shells:     The number of shells to generate.
        deflat:         If True, performs a deflation process. Defaults to False.

    Returns:
        shell verts:    [bs, num_shells, n, 3]
    """
    out_offset_len = offset_len

    if deflat:
        in_offset_len = offset_len

    batch_size = template_verts.shape[0]
    mesh = Meshes(
        verts=template_verts, faces=template_faces[None].repeat(batch_size, 1, 1)
    )
    # bs, n, 3
    vertex_normal = mesh.verts_normals_padded()
    # only for inflating

    if deflat:
        n_inflated_shells = num_shells//2 + 1
    else:
        n_inflated_shells = num_shells
    
    linscale = torch.linspace(
        out_offset_len,
        0,
        n_inflated_shells,
        device=template_verts.device,
        dtype=template_verts.dtype,
    )
    offset = linscale.reshape(1,n_inflated_shells, 1, 1) * vertex_normal[:, None]
    
    if deflat:
        linscale = torch.linspace(0, -in_offset_len, num_shells - n_inflated_shells + 1, device=template_verts.device, dtype=template_verts.dtype)[1:]
        offset_in = linscale.reshape(1, -1, 1, 1) * vertex_normal[:, None]
        offset = torch.cat([offset, offset_in], dim=1)

    verts = template_verts[:, None] + offset
    assert verts.isfinite().all()
    return verts