import numpy as np
from skimage.morphology import skeletonize


# ── L2 functional norm between polynomials ───────────────────────────────────
#
# ||p - q||_{L2[c,d]} = sqrt( integral_c^d (p(x)-q(x))^2 dx )
#
# With r = p - q = sum_k r_k x^k, the integral is the quadratic form r^T G r,
# where G_ij = integral_c^d x^(i+j) dx = (d^(i+j+1) - c^(i+j+1)) / (i+j+1).
# Degree is encoded only by the size of G, so higher-degree structures need no
# new maths — just longer coefficient vectors.
#
# Coefficient convention matches numpy.polyval / PolynomialWithLimits.coeffs:
# DECREASING powers, coeffs[0] is the highest-degree term. Internally we flip
# to increasing powers to index the Gram matrix.


def _gram_matrix(c, d, n_coeff):
    """Gram matrix of monomials {1, x, ..., x^(n_coeff-1)} over [c, d]."""
    powers = np.arange(n_coeff)
    # exponent of the antiderivative for entry (i, j) is i + j + 1
    e = powers[:, None] + powers[None, :] + 1
    return (d ** e - c ** e) / e


def _diff_coeffs_increasing(coeffs_p, coeffs_q):
    """Difference (p - q) as increasing-power coeffs, zero-padded to equal len."""
    p = np.asarray(coeffs_p, dtype=np.float64)[::-1]   # -> increasing powers
    q = np.asarray(coeffs_q, dtype=np.float64)[::-1]
    n = max(p.size, q.size)
    p = np.pad(p, (0, n - p.size))
    q = np.pad(q, (0, n - q.size))
    return p - q


def l2_functional_distance(poly_p, poly_q, interval=None):
    """
    L2 functional distance between two PolynomialWithLimits over an interval.

    interval : (c, d) or None
        Integration interval. If None, uses the ground-truth-style choice of
        poly_q's [xmin, xmax] (call with the ground-truth poly as poly_q).
        Pass an explicit tuple to integrate over a shared/union interval.

    Returns the scalar L2 distance (same units as y, i.e. pixels * sqrt(px)).
    """
    if interval is None:
        c, d = poly_q.xmin, poly_q.xmax
    else:
        c, d = interval
    r = _diff_coeffs_increasing(poly_p.coeffs, poly_q.coeffs)
    G = _gram_matrix(float(c), float(d), r.size)
    val = float(r @ G @ r)
    return np.sqrt(max(val, 0.0))


# torch version for use inside the combined loss (differentiable in coeffs and
# limits). Kept import-light so midline.py is usable without torch installed.

def l2_functional_loss_torch(coeffs_pred, coeffs_gt, c, d):
    """
    Batched, differentiable squared L2 functional norm between predicted and
    ground-truth polynomials over [c, d], summed-then-meaned over the batch.

    coeffs_pred, coeffs_gt : (B, n_coeff) tensors, DECREASING power order.
    c, d                   : (B,) tensors (per-sample integration limits).

    Returns mean over batch of  integral_c^d (p-q)^2 dx  (the SQUARED norm,
    which is the natural smooth quantity to minimise).
    """
    return _integral_sq_torch(coeffs_pred, coeffs_gt, c, d).mean()


# ── two-piece midline loss ───────────────────────────────────────────────────
#
# total = endpoint_term + lambda_int * integral_term
#   endpoint_term : Euclidean distance between matched extreme points
#                   (predicted xmin-point vs GT xmin-point, same for xmax)
#   integral_term : integral_union (p - q)^2 dx over the union of definition
#                   intervals
# Both pieces are differentiable in coeffs and in xmin/xmax. Degree-general:
# the only degree-specific number is n_coeff = degree + 1, used to split the
# flat target vector [coeffs..., xmin, xmax].


try:
    import torch
except ImportError:
    torch = None


def _split_vec_torch(vec, degree):
    """(B, n_coeff+2) -> coeffs (B, n_coeff), xmin (B,), xmax (B,)."""
    n = degree + 1
    return vec[..., :n], vec[..., n], vec[..., n + 1]


def _polyval_torch(coeffs, x):
    """Horner eval, coeffs in DECREASING power order. coeffs (B, n), x (B,)."""
    y = torch.zeros_like(x)
    for k in range(coeffs.shape[-1]):
        y = y * x + coeffs[..., k]
    return y


def _integral_sq_torch(coeffs_pred, coeffs_gt, c, d):
    """Per-sample integral_c^d (p-q)^2 dx via the Gram quadratic form. (B,)."""
    p = torch.flip(coeffs_pred, dims=[-1])
    q = torch.flip(coeffs_gt, dims=[-1])
    r = p - q
    n = r.shape[-1]
    powers = torch.arange(n, device=r.device, dtype=r.dtype)
    e = powers[:, None] + powers[None, :] + 1
    c = c.to(r.dtype).reshape(-1, 1, 1)
    d = d.to(r.dtype).reshape(-1, 1, 1)
    G = (d ** e - c ** e) / e
    return torch.clamp(torch.einsum('bi,bij,bj->b', r, G, r), min=0.0)


def midline_loss_torch(pred, gt, degree=1, lambda_int=1.0, eps=1e-8):
    """
    Two-piece midline loss between predicted and ground-truth target vectors.

    pred, gt : (B, degree+3) tensors = [coeffs (decreasing powers), xmin, xmax].
               For degree 1: [a, b, xmin, xmax].

    Returns (total, endpoint_term, integral_term) as scalar tensors (each
    already meaned over the batch) so the two pieces can be logged separately.
    """
    import torch
    cp, xminp, xmaxp = _split_vec_torch(pred, degree)
    cg, xming, xmaxg = _split_vec_torch(gt, degree)

    # piece 1: matched extreme-point Euclidean distances
    yminp, ymaxp = _polyval_torch(cp, xminp), _polyval_torch(cp, xmaxp)
    yming, ymaxg = _polyval_torch(cg, xming), _polyval_torch(cg, xmaxg)
    left  = torch.sqrt((xminp - xming) ** 2 + (yminp - yming) ** 2 + eps)
    right = torch.sqrt((xmaxp - xmaxg) ** 2 + (ymaxp - ymaxg) ** 2 + eps)
    endpoint_term = (left + right).mean()

    # piece 2: integral over the union of definition intervals
    c = torch.minimum(xminp, xming)
    d = torch.maximum(xmaxp, xmaxg)
    integral_term = _integral_sq_torch(cp, cg, c, d).mean()

    total = endpoint_term + lambda_int * integral_term
    return total, endpoint_term, integral_term


# ── label name handling ──────────────────────────────────────────────────────

MIDLINE_CANONICAL = 'Midline'


def _normalise(name):
    return name.strip().lower().replace(' ', '').replace('_', '')


def find_midline_layer(layer_names):
    """
    Locate the midline layer among XCF layer names, accepting capitalisation /
    spacing / underscore variants of 'Midline'. Returns the matching raw name.

    Raises ValueError if there is no match, or if there are several plausible
    matches (so a wrongly-named layer is reported rather than silently used).
    """
    target = _normalise(MIDLINE_CANONICAL)
    matches = [n for n in layer_names if _normalise(n) == target]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        # surface near-misses to help diagnose a misnamed layer
        near = [n for n in layer_names if 'mid' in _normalise(n)]
        hint = ' Possible misnamed candidates: {:}.'.format(near) if near else ''
        raise ValueError(
            "No '{:}' layer found (accepting caps/space/underscore variants)."
            "{:}".format(MIDLINE_CANONICAL, hint)
        )
    raise ValueError(
        "Ambiguous midline layer: several names match '{:}': {:}".format(
            MIDLINE_CANONICAL, matches
        )
    )


# ── polynomial with limits ───────────────────────────────────────────────────

class PolynomialWithLimits:
    """
    An explicit polynomial y = c[0]*x^(d) + ... + c[d] valid on [xmin, xmax].

    For the midline this is degree 1: y = a*x + b, coeffs = [a, b].
    Designed to generalise to higher-degree structures later: the only
    degree-specific assumption is in the fitting helpers, not in the container.

    The target vector exposed to the network is: list(coeffs) + [xmin, xmax].
    For degree 1 that is the requested 4 values [a, b, xmin, xmax].
    """

    def __init__(self, coeffs, xmin, xmax):
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.degree = len(self.coeffs) - 1
        self.xmin = float(xmin)
        self.xmax = float(xmax)

    # -- evaluation --
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        return np.polyval(self.coeffs, x)

    def endpoints(self):
        """Return ((xmin, y(xmin)), (xmax, y(xmax)))."""
        return ((self.xmin, float(self(self.xmin))),
                (self.xmax, float(self(self.xmax))))

    # -- (de)serialisation to the flat network target --
    def to_vector(self):
        """coeffs followed by [xmin, xmax]."""
        return np.concatenate([self.coeffs, [self.xmin, self.xmax]]).astype(np.float32)

    @classmethod
    def from_vector(cls, vec, degree=1):
        vec = np.asarray(vec, dtype=np.float64)
        n_coeff = degree + 1
        coeffs = vec[:n_coeff]
        xmin, xmax = vec[n_coeff], vec[n_coeff + 1]
        return cls(coeffs, xmin, xmax)

    def __repr__(self):
        return 'PolynomialWithLimits(coeffs={:}, xmin={:.2f}, xmax={:.2f})'.format(
            self.coeffs.tolist(), self.xmin, self.xmax
        )


# ── midline extraction ───────────────────────────────────────────────────────

def midline_from_mask(binary_mask, degree=1):
    """
    Build a PolynomialWithLimits from a binary midline mask.

    Procedure (degree 1):
      1. skeletonise the binary mask
      2. find the skeleton's leftmost x (xmin) and rightmost x (xmax)
      3. take the skeleton point at xmin and the skeleton point at xmax
         (averaging y if several skeleton pixels share that x)
      4. the line is the one passing through those two endpoints, returned
         in explicit form y = a*x + b on [xmin, xmax]

    Note: per the spec the line is defined by the endpoints at xmin/xmax, NOT a
    least-squares fit over all skeleton points.

    Returns None if the mask is empty or degenerate (all skeleton pixels share
    a single x, so no line in explicit y=ax+b form exists).
    """
    mask = np.asarray(binary_mask) > 0
    if not mask.any():
        return None

    skel = skeletonize(mask)
    ys, xs = np.nonzero(skel)
    if xs.size == 0:
        return None

    xmin = int(xs.min())
    xmax = int(xs.max())
    if xmin == xmax:
        # vertical / single-column skeleton: not expressible as y = a*x + b
        return None

    y_at_xmin = float(ys[xs == xmin].mean())
    y_at_xmax = float(ys[xs == xmax].mean())

    if degree == 1:
        a = (y_at_xmax - y_at_xmin) / (xmax - xmin)
        b = y_at_xmin - a * xmin
        coeffs = [a, b]
    else:
        # placeholder for future higher-degree structures: fit through the
        # skeleton points constrained to the endpoint x-range.
        coeffs = np.polyfit(xs, ys, degree)

    return PolynomialWithLimits(coeffs, xmin, xmax)

