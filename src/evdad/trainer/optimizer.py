import torch


def get_optimizer(cfg: dict, net: torch.nn.Module) -> torch.optim.Optimizer:
    """This function instantiates optimizer based on givem config.

    Args:
        cfg: Config obtained from hydra plugin.
        net: Network model.

    Returns:
        Optimizer
    """
    optim_type = cfg["optimizer"]["type"]
    lr = cfg["optimizer"]["lr"]

    if optim_type == "adam":
        weight_decay = cfg["optimizer"]["weight_decay"]
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == "adamw":
        weight_decay = cfg["optimizer"]["weight_decay"]
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError("This optimizer is not implemented")
