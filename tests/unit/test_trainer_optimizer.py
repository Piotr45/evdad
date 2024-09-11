from torch.optim import Adam, AdamW

from evdad.models.lava.slayer.cuba.simple_dense import SimpleDense
from evdad.trainer.optimizer import get_optimizer


def test_trainer_optimizer():
    cfg_adam = {"optimizer": {"type": "adam", "lr": 1e-3, "weight_decay": 1e-4}}
    cfg_adamw = {"optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 1e-4}}
    model = SimpleDense(2, 32, 32, 10, 0.1, 0.1, 0.1, 1, 1, False)

    optimizer = get_optimizer(cfg_adam, model)
    assert isinstance(optimizer, Adam)
    optimizer = get_optimizer(cfg_adamw, model)
    assert isinstance(optimizer, AdamW)
