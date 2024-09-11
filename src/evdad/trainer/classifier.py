import lava.lib.dl.slayer as slayer
import torch


def get_classifier(cfg: dict) -> torch.nn.Module:
    """Get classifier based on given config.

    Args:
        cfg: Hydra config.

    Returns:
        Classifier function.
    """
    if cfg["loss"]["loss_function"] == "SpikeRate":
        return slayer.classifier.Rate.predict
    return None
