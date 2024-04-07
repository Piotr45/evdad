import lava.lib.dl.slayer as slayer
import torch


def get_classifier(cfg: dict) -> torch.nn.Module:
    """Get calssifier based on given config.

    Arfs:
        cfg: Hydra config.

    Returns:
        Classifier function.
    """
    if cfg["loss"]["loss_function"] == "SpikeRate":
        return slayer.classifier.Rate.predict
    return None
