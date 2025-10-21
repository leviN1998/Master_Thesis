import torch
import roma



class RotationLoss(torch.nn.Module):
    def __init__(self, lam=1e-3):
        super(RotationLoss, self).__init__()
        self.lam = lam


    def old_loss(self, M_pred, R_gt):
        u, s, v = torch.linalg.svd(M_pred)
        det = torch.linalg.det(u @ v.transpose(-2, -1))
        I = torch.eye(3, dtype=M_pred.dtype, device=M_pred.device).unsqueeze(0).repeat(M_pred.shape[0], 1, 1)
        dia = I.clone()
        dia[:, 2, 2] = det
        R = u @ dia @ v.transpose(-2, -1)
        R = roma.special_procrustes(R)

        R_hat = R
        # print(R_hat.shape, R_gt.shape)
        loss_mat = torch.linalg.matrix_norm(R_hat - R_gt, ord='fro')**2
        proj = torch.linalg.matrix_norm(R_hat - M_pred, ord='fro')**2
        
        return loss_mat.mean() + self.lam * proj.mean(), R_hat
    
    def geodesic_loss(self, M_pred, R_gt, reduce=True):
        R_pred = procrustes_to_rotmat(M_pred)
        M = R_pred.transpose(-1, -2) @ R_gt
        cos_theta = (M.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1.0) * 0.5
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0- 1e-6)
        theta = torch.arccos(cos_theta)
        loss = theta**2
        if reduce:
            loss = loss.mean()
        return loss, R_pred
    
    def vector_loss(self, v_pred, v_gt, w_dir=1.0, w_mag=0.0):
        eps = 1e-8

        pred_u = v_pred / (v_pred.norm(dim=-1, keepdim=True) + eps)
        target_u = v_gt / (v_gt.norm(dim=-1, keepdim=True) + eps)
        dir_loss = (1.0 - (pred_u * target_u).sum(dim=-1)).mean()

        cos_loss = (pred_u * target_u).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        dir_loss = (1 - cos_loss).mean()

        mag_loss = torch.nn.functional.smooth_l1_loss(
            v_pred.norm(dim=-1), v_gt.norm(dim=-1)
        )

        return w_dir * dir_loss + w_mag * mag_loss, v_pred

    def forward(self, M_pred, R_gt):
        #loss, R_pred = self.geodesic_loss(M_pred, R_gt)
        loss, R_pred = self.vector_loss(M_pred, R_gt)

        return loss, R_pred
        
    

def procrustes_to_rotmat(inp: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 3x3 matrices to the nearest rotation matrices using SVD.

    Args:
        inp: A tensor of shape (B, 3, 3) representing a batch of 3x3 matrices.

    Returns:
        A tensor of shape (B, 3, 3) representing the nearest rotation matrices.
    """
    orig_shape = inp.shape
    m = inp.reshape(-1, 3, 3)
    U, S, Vh = torch.linalg.svd(m)

    det_uvt = torch.linalg.det(U @ Vh)
    sgn = torch.where(det_uvt > 0,
                      torch.ones_like(det_uvt),
                      -torch.ones_like(det_uvt))

    dvec = torch.stack([torch.ones_like(sgn), torch.ones_like(sgn), sgn], dim=-1)
    D = torch.diag_embed(dvec)
    R = U @ D @ Vh
    return R.view(orig_shape)

    

def skew(a):
    # a: (...,3) unit axis
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    K = torch.zeros(a.shape[:-1] + (3,3), dtype=a.dtype, device=a.device)
    K[..., 0,1] = -az; K[..., 0,2] =  ay
    K[..., 1,0] =  az; K[..., 1,2] = -ax
    K[..., 2,0] = -ay; K[..., 2,1] =  ax
    return K

@torch.no_grad()
def is_rotmat(R, tol=1e-3):
    I = torch.eye(3, dtype=R.dtype, device=R.device)
    ortho = torch.linalg.matrix_norm(R.transpose(-1,-2) @ R - I, ord='fro')
    det = torch.linalg.det(R)
    return ortho.max().item() < tol and torch.allclose(det, torch.ones_like(det), atol=1e-3)

def axis_angle_to_rotmat_torch(axis_angle, eps=1e-8):
    """
    axis_angle: (...,3), entries in radians; vector = axis * angle
    returns:    (...,3,3) proper rotation matrices (SO(3))
    """
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)                 # (...,1)
    axis  = axis_angle / torch.clamp(theta, min=eps)                            # unit axis
    K = skew(axis)                                                              # (...,3,3)
    I = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(axis.shape[:-1]+(3,3))
    sin_t = torch.sin(theta)[..., None]                                         # (...,1,1)
    cos_t = torch.cos(theta)[..., None]                                         # (...,1,1)

    # Rodrigues: R = I + sinθ K + (1−cosθ) K^2  (requires ||axis||=1)
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)

    # Handle tiny angles to avoid numerical noise (optional but nice)
    tiny = (theta.squeeze(-1) < 1e-6)[..., None, None]
    if tiny.any():
        # 2nd-order Taylor: I + Kθ + 0.5 K^2 θ^2
        t = theta[..., None]                                                    # (...,1,1)
        R_taylor = I + t * K + 0.5 * (t * t) * (K @ K)
        R = torch.where(tiny, R_taylor, R)
    return R


def angle_error_deg(R_pred, R_gt):
    """Compute the angular error in degrees between two rotation matrices.

    Args:
        R_pred: Tensor of shape (..., 3, 3) representing predicted rotation matrices.
        R_gt: Tensor of shape (..., 3, 3) representing ground truth rotation matrices.
    Returns:
        Tensor of shape (...) representing the angular error in degrees.
    """
    # temp for simple loss
    #print("angle_error:", R_pred.shape, R_gt.shape)
    p = R_pred / (R_pred.norm(dim=-1, keepdim=True) + 1e-8)
    t = R_gt / (R_gt.norm(dim=-1, keepdim=True) + 1e-8)
    cos = torch.sum(p * t, dim=1)
    cos = torch.clamp(cos, -1.0+1e-7, 1.0-1e-7)
    angle_rad = torch.acos(cos)
    angle_deg = torch.rad2deg(angle_rad)
    return angle_deg


    # print("angle_error:", R_pred.shape, R_gt.shape)
    tr = torch.einsum("bij,bji->b", R_pred.transpose(-1,-2), R_gt)
    x = torch.clamp(0.5*(tr - 1.0), -1.0, 1.0)
    return torch.rad2deg(torch.arccos(x))      # (B,)

def acc_at_threshold_deg(theta_deg, thr=5.0):
    return (theta_deg <= thr).float().mean()


"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hilfsfunktion: zufällige Achse-Winkel-Rotation erzeugen
def random_axis_angle(n):
    axis = torch.randn(n, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.rand(n, 1, device=device) * 2 * torch.pi  # 0..2π
    return axis * angle  # (n,3)

# Rodrigues-Funktion aus deinem Code:
def skew(a):
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    K = torch.zeros(a.shape[:-1] + (3, 3), dtype=a.dtype, device=a.device)
    K[..., 0, 1] = -az; K[..., 0, 2] = ay
    K[..., 1, 0] = az;  K[..., 1, 2] = -ax
    K[..., 2, 0] = -ay; K[..., 2, 1] = ax
    return K

def axis_angle_to_rotmat(axis_angle, eps=1e-8):
    theta = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / torch.clamp(theta, min=eps)
    K = skew(axis)
    I = torch.eye(3, device=axis.device).expand(axis.shape[:-1] + (3, 3))
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    return R


B = 10
axis_angle = random_axis_angle(B)
R_gt = axis_angle_to_rotmat(axis_angle)

print("GT shapes:", R_gt.shape)
print("Determinants:", torch.linalg.det(R_gt))


M = torch.nn.Parameter(torch.randn(B, 3, 3, device=device) * 0.01)
opt = torch.optim.Adam([M], lr=1e-2)
loss = RotationLoss()

for t in range(2000):
    opt.zero_grad()
    l, R_pred = loss(M, R_gt)
    l.backward()
    opt.step()
    if t % 100 == 0:
        print(f"Step {t}, Loss: {l.item():.6f}")
"""