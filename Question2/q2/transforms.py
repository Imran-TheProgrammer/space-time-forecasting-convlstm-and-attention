"""Mirror, rotation and the 8 symmetries of the square. All act on the last two axes."""
import torch


def mirror(x):
    """(Mx)[i, j] = x[i, 63 - j]."""
    return torch.flip(x, dims=(-1,))


def rotate(x, a=1):
    """R^a: rotate by a * 90 degrees anticlockwise."""
    return torch.rot90(x, k=a, dims=(-2, -1))


# Each element g = (m, a) means M^m R^a: rotate by a first, then mirror if m = 1.
GROUP = [(m, a) for m in (0, 1) for a in range(4)]


def apply(x, g):
    m, a = g
    y = rotate(x, a)
    return mirror(y) if m else y


def invert(x, g):
    """Apply g^{-1}: R^a has inverse R^{-a}; each M R^a is its own inverse."""
    m, a = g
    return apply(x, g) if m else rotate(x, -a)


def apply_per_video(x, g_index):
    """Apply a different symmetry GROUP[g_index[b]] to each video b of x [B, ...]."""
    out = torch.empty_like(x)
    for i, g in enumerate(GROUP):
        chosen = (g_index == i).nonzero(as_tuple=True)[0]
        if len(chosen):
            out[chosen] = apply(x[chosen], g)
    return out
