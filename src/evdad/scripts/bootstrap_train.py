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


def bootstrap_training_loop(
    net: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: Union[DataLoader, None],
    mlflow_logger: MlflowClient,
    stats: slayer.utils.LearningStats,
    error: Any,
    optimizer: Any,
    skip_test: bool,
    epochs: int,
) -> None:
    scheduler = bootstrap.routine.Scheduler(num_sample_iter=10, sample_period=10)
    device = get_device()

    for epoch in tqdm.tqdm(
        range(0, epochs),
        desc="Epochs",
    ):
        for i, (input, target) in enumerate(train_loader, 0):
            net.train()
            mode = scheduler.mode(epoch, i, net.training)

            input = input.to(device)

            output, count = net.forward(input, mode)

            loss = error(output, target.to(device))

            prediction = output.data.max(1, keepdim=True)[1].cpu().flatten()
            # prediction = torch.squeeze(output.data.max(1, keepdim=True)[1].cpu())

            stats.training.num_samples += len(target)
            # stats.training.num_samples += len(label.flatten())
            stats.training.loss_sum += loss.cpu().data.item() * input.shape[0]
            stats.training.correct_samples += torch.sum(prediction == target).data.item()
            # stats.training.correct_samples += torch.sum(prediction == label.flatten()).data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            header = [str(mode)]
            header += ["Event rate : " + ", ".join([f"{c.item():.4f}" for c in count.flatten()])]
            stats.print(epoch + 1, iter=i, header=header, dataloader=train_loader)

        mlflow_logger.log_metric("training_loss", stats.training.loss, step=epoch + 1)
        mlflow_logger.log_metric("training_accuracy", stats.training.accuracy, step=epoch + 1)

        if not skip_test:
            for i, (input, target) in enumerate(test_loader, 0):
                net.eval()
                mode = scheduler.mode(epoch, i, net.training)

                with torch.no_grad():
                    input = input.to(device)

                    output, count = net.forward(input, mode=scheduler.mode(epoch, i, net.training))

                    loss = error(output, target.to(device))
                    prediction = output.data.max(1, keepdim=True)[1].cpu().flatten()

                stats.testing.num_samples += len(target)
                # stats.testing.num_samples += len(label.flatten())
                stats.testing.loss_sum += loss.cpu().data.item() * input.shape[0]
                stats.testing.correct_samples += torch.sum(prediction == target).data.item()
                # stats.testing.correct_samples += torch.sum(prediction == label.flatten()).data.item()

                header = [str(mode)]
                header += ["Event rate : " + ", ".join([f"{c.item():.4f}" for c in count.flatten()])]
                stats.print(epoch + 1, iter=i, header=header, dataloader=test_loader)

            mlflow_logger.log_metric("test_loss", stats.testing.loss, step=epoch + 1)
            mlflow_logger.log_metric("test_accuracy", stats.testing.accuracy, step=epoch + 1)

        save_new_checkpoint(net, epoch + 1, mlflow_logger.run_id)

        if stats.testing.best_accuracy:
            checkpoint_data = {"checkpoint": net.state_dict(), "epoch": epoch + 1, "run_id": mlflow_logger.run_id}
            torch.save(
                checkpoint_data,
                os.path.join(get_hydra_dir_path(), "checkpoint_best.pt"),
            )

        mlflow_logger.log_artifact(get_logger_path("bootstrap_train.log"))
        stats.update()
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

    mlflow_logger = hydra.utils.instantiate(cfg["mlflow"])

    mlflow_logger.run_experiment()

    mlflow_logger.log_param("epochs", epochs)
    mlflow_logger.log_param("batch size", batch_size)
    mlflow_logger.log_param("learning rate", lr)

    mlflow_logger.log_artifact(get_logger_path("bootstrap_train.log"))

    torch.cuda.empty_cache()
    bootstrap_training_loop(
        net=net,
        train_loader=train_loader,
        test_loader=test_loader if not skip_test else None,
        mlflow_logger=mlflow_logger,
        stats=stats,
        error=error,
        optimizer=optimizer,
        skip_test=skip_test,
        epochs=epochs,
    )

    mlflow.pytorch.log_model(net, mlflow_logger.run_id)
    mlflow_logger.finish_experiment()
    print("Model training and logging complete.")


if __name__ == "__main__":
    main()
