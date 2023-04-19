import torch
import torch.nn as nn

def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )

def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)

def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.
    
    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R


def get_node_repr(coords, gvp_out_scalars, gvp_out_vectors):
    R = get_rotation_frames(coords[:,:3,:])
    # Rotate to local rotation frame for rotation-invariance
    gvp_out_features = torch.cat([
        gvp_out_scalars,
        rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
    ], dim=-1)

    return gvp_out_features

if __name__ == "__main__":
    node_hidden_dim_scalar = 100
    node_hidden_dim_vector = 16
    embed_dim = 512

    gvp_out_dim = node_hidden_dim_scalar + (3 *node_hidden_dim_vector)
    embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

    coords = torch.randn(234, 4, 3)
    gvp_out_scalars = torch.randn(234, node_hidden_dim_scalar)
    gvp_out_vectors = torch.randn(234, node_hidden_dim_vector, 3)


    gvp_out_features = get_node_repr(coords, gvp_out_scalars, gvp_out_vectors)
    # R = get_rotation_frames(coords)
    # # Rotate to local rotation frame for rotation-invariance
    # gvp_out_features = torch.cat([
    # 	gvp_out_scalars,
    # 	rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
    # ], dim=-1)
    gvp_out = embed_gvp_output(gvp_out_features)

    print(gvp_out.shape)