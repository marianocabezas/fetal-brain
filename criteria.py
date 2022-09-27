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
