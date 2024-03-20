import logging
import os

import hydra
import lava.lib.dl.slayer as slayer
import mlflow
import setproctitle
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from evdad.trainer.loss import get_loss_function
from evdad.trainer.optimizer import get_optimizer
from evdad.utils import get_device, get_hydra_dir_path, get_logger_path

log = logging.getLogger(__name__)


def save_new_checkpoint(net: torch.nn.Module, epoch: int) -> None:
    """Saves new checkpoint and removes previous one."""
    torch.save(net.state_dict(), os.path.join(get_hydra_dir_path(), f"checkpoint_{epoch}.pt"))

    last_checkpoint_path = os.path.join(get_hydra_dir_path(), f"checkpoint_{epoch-1}.pt")
    if os.path.exists(last_checkpoint_path):
        os.remove(last_checkpoint_path)


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    setproctitle.setproctitle("evdad-train")

    log.info(OmegaConf.to_yaml(cfg))

    device = get_device()
    log.info(f"Using device: {device}")

    dataset = hydra.utils.instantiate(cfg["dataset"])

    batch_size = cfg["dataloader"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["optimizer"]["lr"]
    skip_test = cfg["training"]["skip_test"]

    train = dataset.get_train_dataset()
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2)

    if not skip_test:
        test = dataset.get_test_dataset()
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True, num_workers=0, prefetch_factor=0)

    net = hydra.utils.instantiate(cfg["model"]).to(device)

    optimizer = get_optimizer(cfg, net)
    error = get_loss_function(cfg, device)

    stats = slayer.utils.LearningStats()

    assistant = slayer.utils.Assistant(
        net=net,
        error=error,
        optimizer=optimizer,
        stats=stats,
        # error=lambda output, target: torch.nn.functional.mse_loss(output, target),
        # classifier=slayer.classifier.Rate.predict,
    )

    mlflow_logger = hydra.utils.instantiate(cfg["mlflow"])

    mlflow_logger.run_experiment()

    mlflow_logger.log_param("epochs", epochs)
    mlflow_logger.log_param("batch size", batch_size)
    mlflow_logger.log_param("learning rate", lr)

    mlflow_logger.log_artifact(get_logger_path("train.log"))

    torch.cuda.empty_cache()
    for epoch in tqdm.tqdm(
        range(epochs),
        desc="Epochs",
    ):
        for i, (input, target) in enumerate(train_loader):  # training loop
            output = assistant.train(
                input.to(device, dtype=torch.float),
                target.to(device, dtype=torch.float),
            )
            stats.print(epoch, iter=i, dataloader=train_loader)

        mlflow_logger.log_metric("training_loss", stats.training.loss, step=epoch)
        mlflow_logger.log_metric("training_accuracy", stats.training.accuracy, step=epoch)

        if not skip_test:
            for i, (input, target) in enumerate(test_loader):  # test loop
                output = assistant.test(input.to(device, dtype=torch.float), target.to(device, dtype=torch.float))
                stats.print(epoch, iter=i, dataloader=test_loader)

            mlflow_logger.log_metric("test_loss", stats.testing.loss, step=epoch)
            mlflow_logger.log_metric("test_accuracy", stats.testing.accuracy, step=epoch)

        save_new_checkpoint(net, epoch)

        if stats.training.best_loss:
            torch.save(net.state_dict(), os.path.join(get_hydra_dir_path(), "checkpoint_best.pt"))
            # mlflow.pytorch.log_model(net, mlflow_logger.run_id) # TODO fix warnings

        mlflow_logger.log_artifact(get_logger_path("train.log"))
        stats.update()

    mlflow.pytorch.log_model(net, mlflow_logger.run_id)
    mlflow_logger.finish_experiment()
    print("Model training and logging complete.")


if __name__ == "__main__":
    main()
