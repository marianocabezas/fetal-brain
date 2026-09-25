import torch


def dsc_binary_loss(pred, target, logits=True):
    if logits:
        pred = torch.sigmoid(pred)
    pred = torch.flatten(pred >= 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
            2 * torch.sum(pred & target, dim=1)
    ).type(torch.float32).to(pred.device)
    sum_pred = torch.sum(pred, dim=1).type(torch.float32).to(pred.device)
    sum_target = torch.sum(target, dim=1).type(torch.float32).to(pred.device)

    dsc_k = intersection / (sum_pred + sum_target)
    dsc_k = dsc_k[torch.logical_not(torch.isnan(dsc_k))]
    if len(dsc_k) > 0:
        dsc = 1 - torch.mean(dsc_k)
    else:
        dsc = torch.tensor(0)

    return torch.clamp(dsc, 0., 1.)


def tp_binary_loss(pred, target, logits=True):
    if logits:
        pred = torch.sigmoid(pred)
    pred = torch.flatten(pred >= 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
        torch.sum(pred & target, dim=1)
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(target, dim=1).type(torch.float32).to(pred.device)

    tp_k = intersection / sum_target
    tp_k = tp_k[torch.logical_not(torch.isnan(tp_k))]
    tp = 1 - torch.mean(tp_k) if len(tp_k) > 0 else torch.tensor(0)

    return torch.clamp(tp, 0., 1.)


def tn_binary_loss(pred, target, logits=True):
    if logits:
        pred = torch.sigmoid(pred)
    pred = torch.flatten(pred < 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
            torch.sum(pred & torch.logical_not(target), dim=1)
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(
        torch.logical_not(target), dim=1
    ).type(torch.float32).to(pred.device)

    tn_k = intersection / sum_target
    tn_k = tn_k[torch.logical_not(torch.isnan(tn_k))]
    tn = 1 - torch.mean(tn_k)
    if torch.isnan(tn):
        tn = torch.tensor(0.)

    return torch.clamp(tn, 0., 1.)


def ellipse_loss(prediction, target, num_steps=10000):
    """
    Calculates the integrated L2 distance between two ellipses.
    Note: An analytical closed-form solution does not exist for general ellipses
    due to the nature of elliptic integrals. This vectorized approach is the
    standard differentiable method for optimization.
    """
    device = prediction.device
    t = torch.linspace(0, 2 * np.pi, num_steps, device=device)

    e1_params = {
        'x0': prediction[:, 0],
        'y0': prediction[:, 1],
        'cos_theta': prediction[:, 2],
        'sin_theta': prediction[:, 3],
        'a': prediction[:, 4],
        'b': prediction[:, 4] * prediction[:, 5],
    }

    e2_params = {
        'x0': target[:, 0],
        'y0': target[:, 1],
        'cos_theta': target[:, 2],
        'sin_theta': target[:, 3],
        'a': target[:, 4],
        'b': target[:, 4] * target[:, 5],
    }

    def get_points(p):
        cos_t = torch.cos(t).unsqueeze(0)
        sin_t = torch.sin(t).unsqueeze(0)
        cos_theta = p['cos_theta'].unsqueeze(1)
        sin_theta = p['sin_theta'].unsqueeze(1)

        # Standard parametric form
        x = p['a'].unsqueeze(1) * cos_t
        y = p['b'].unsqueeze(1) * sin_t

        # Rotation and translation
        xr = p['x0'].unsqueeze(1) + x * cos_theta - y * sin_theta
        yr = p['y0'].unsqueeze(1) + x * sin_theta + y * cos_theta
        return xr, yr

    x1, y1 = get_points(e1_params)
    x2, y2 = get_points(e2_params)

    # Differentiable L2 distance
    dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + 1e-8)

    # Mean distance scaled by 2*pi represents the integral over t
    return torch.mean(dist) * (2 * torch.pi)


def line_loss(prediction, target):
    e1_params = {
        'x0': prediction[:, 0],
        'y0': prediction[:, 1],
        'cos_theta': prediction[:, 2],
        'sin_theta': prediction[:, 3],
        'a': prediction[:, 4],
        'b': prediction[:, 4] * prediction[:, 5],
    }

    e2_params = {
        'x0': target[:, 0],
        'y0': target[:, 1],
        'cos_theta': target[:, 2],
        'sin_theta': target[:, 3],
        'a': target[:, 4],
        'b': target[:, 4] * target[:, 5],
    }

    Px = e1_params['x0'] - e2_params['x0']
    Py = e1_params['y0'] - e2_params['y0']

    Vx = e1_params['cos_theta'] - e2_params['cos_theta']
    Vy = e1_params['sin_theta'] - e2_params['sin_theta']

    # We integrate over the differences in the [0, 1] interval.
    # Defining a large interval would only serve as weighting mechanism.

    x_2 = (Vx ** 2 + Vy ** 2) / 3
    x = Px * Vx + Py * Vy
    c = Px ** 2 + Py ** 2

    return torch.mean(x_2 + x + c)
