import logging
import os
from typing import Any, Union

import hydra
import lava.lib.dl.bootstrap as bootstrap
import lava.lib.dl.slayer as slayer
import mlflow
import setproctitle
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from evdad.metrics.mlflow import MlflowClient
from evdad.trainer.classifier import get_classifier
from evdad.trainer.loss import get_loss_function
from evdad.trainer.optimizer import get_optimizer
from evdad.utils import get_device, get_hydra_dir_path, get_logger_path

log = logging.getLogger(__name__)


def save_new_checkpoint(net: torch.nn.Module, epoch: int, run_id: str) -> None:
    """Saves new checkpoint and removes previous one."""
    checkpoint_data = {"checkpoint": net.state_dict(), "epoch": epoch, "run_id": run_id}
    torch.save(checkpoint_data, os.path.join(get_hydra_dir_path(), f"checkpoint_{epoch}.pt"))

    last_checkpoint_path = os.path.join(get_hydra_dir_path(), f"checkpoint_{epoch-1}.pt")
    if os.path.exists(last_checkpoint_path):
        os.remove(last_checkpoint_path)


def slayer_training_loop(
    net: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: Union[DataLoader, None],
    mlflow_logger: MlflowClient,
    stats: slayer.utils.LearningStats,
    assistant: slayer.utils.Assistant,
    skip_test: bool,
    epochs: int,
) -> None:
    device = get_device()
    for epoch in tqdm.tqdm(
        range(1, epochs + 1),
        desc="Epochs",
    ):
        for i, (input, target) in enumerate(train_loader):  # training loop
            _, count = assistant.train(
                input.to(device, dtype=torch.float),
                target.to(device),  # target.to(device, dtype=torch.float),
            )
            header = ["Event rate: " + ", ".join([f"{c.item():.4f}" for c in count.flatten()])]
            stats.print(epoch, iter=i, dataloader=train_loader, header=header)

        # log.info(
        #     f"Epoch: {epoch}\tTraining loss: {stats.training.loss}\tTraining accuracy: {stats.training.accuracy}\t{header[0]}\tEvent input: {torch.mean(target).item()}"
        # )

        mlflow_logger.log_metric("training_loss", stats.training.loss, step=epoch)
        mlflow_logger.log_metric("training_accuracy", stats.training.accuracy, step=epoch)

        if not skip_test:
            for i, (input, target) in enumerate(test_loader):  # test loop
                _, count = assistant.test(
                    input.to(device, dtype=torch.float),
                    target.to(device),  # target.to(device, dtype=torch.float),
                )
                stats.print(epoch, iter=i, dataloader=test_loader)

            # log.info(f"Epoch: {epoch}\tTest loss: {stats.testing.loss}\tTest accuracy: {stats.testing.accuracy}")

            mlflow_logger.log_metric("test_loss", stats.testing.loss, step=epoch)
            mlflow_logger.log_metric("test_accuracy", stats.testing.accuracy, step=epoch)

        save_new_checkpoint(net, epoch, mlflow_logger.run_id)

        if stats.testing.best_loss:
            checkpoint_data = {"checkpoint": net.state_dict(), "epoch": epoch, "run_id": mlflow_logger.run_id}
            torch.save(
                checkpoint_data,
                os.path.join(get_hydra_dir_path(), "checkpoint_best.pt"),
            )
            # mlflow.pytorch.log_model(net, mlflow_logger.run_id) # TODO fix warnings

        mlflow_logger.log_artifact(get_logger_path("slayer_train.log"))
        stats.update()

    net.export_hdf5(f"checkpoint_last.net")
    return


@hydra.main(version_base=None, config_path="../../conf/experiments")
def main(cfg: DictConfig) -> None:
    setproctitle.setproctitle("evdad-train")

    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Output will be saved at {get_hydra_dir_path()}")

    device = get_device()
    log.info(f"Using device: {device}")

    dataset = hydra.utils.instantiate(cfg["dataset"])

    batch_size = cfg["dataloader"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["optimizer"]["lr"]
    skip_test = cfg["training"]["skip_test"]
    num_workers = cfg["dataloader"]["num_workers"]

    train = dataset.get_train_dataset()
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    if not skip_test:
        test = dataset.get_test_dataset()
        test_loader = DataLoader(
            dataset=test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    net = hydra.utils.instantiate(cfg["model"]).to(device)

    log.info(net)

    optimizer = get_optimizer(cfg, net)
    error = get_loss_function(cfg, device)

    stats = slayer.utils.LearningStats()

    assistant = slayer.utils.Assistant(
        net=net,
        error=error,
        optimizer=optimizer,
        stats=stats,
        classifier=get_classifier(cfg),
        count_log=True,
    )

    mlflow_logger = hydra.utils.instantiate(cfg["mlflow"])

    mlflow_logger.run_experiment()

    mlflow_logger.log_param("epochs", epochs)
    mlflow_logger.log_param("batch size", batch_size)
    mlflow_logger.log_param("learning rate", lr)

    mlflow_logger.log_artifact(get_logger_path("slayer_train.log"))

    torch.cuda.empty_cache()
    slayer_training_loop(
        net=net,
        train_loader=train_loader,
        test_loader=test_loader if not skip_test else None,
        mlflow_logger=mlflow_logger,
        stats=stats,
        assistant=assistant,
        skip_test=skip_test,
        epochs=epochs,
    )

    mlflow.pytorch.log_model(net, mlflow_logger.run_id)
    mlflow_logger.finish_experiment()
    print("Model training and logging complete.")


if __name__ == "__main__":
    main()
