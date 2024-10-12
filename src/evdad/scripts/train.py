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
    """TODO"""
    device = get_device()
    for epoch in tqdm.tqdm(
        range(1, epochs + 1),
        desc="Epochs",
    ):
        for i, (input, target) in enumerate(train_loader):  # training loop
            _, count = assistant.train(
                input.to(device, dtype=torch.float),
                target,  # target.to(device, dtype=torch.float),
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
                    target,  # target.to(device, dtype=torch.float),
                )
                stats.print(epoch, iter=i, dataloader=test_loader)

            # log.info(f"Epoch: {epoch}\tTest loss: {stats.testing.loss}\tTest accuracy: {stats.testing.accuracy}")

            mlflow_logger.log_metric("test_loss", stats.testing.loss, step=epoch)
            mlflow_logger.log_metric("test_accuracy", stats.testing.accuracy, step=epoch)

        save_new_checkpoint(net, epoch, mlflow_logger.run_id)

        if stats.testing.best_accuracy:
            checkpoint_data = {"checkpoint": net.state_dict(), "epoch": epoch, "run_id": mlflow_logger.run_id}
            torch.save(
                checkpoint_data,
                os.path.join(get_hydra_dir_path(), "checkpoint_best.pt"),
            )
            # mlflow.pytorch.log_model(net, mlflow_logger.run_id) # TODO fix warnings

        mlflow_logger.log_artifact(get_logger_path("train.log"))
        stats.update()
    return


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

        mlflow_logger.log_artifact(get_logger_path("train.log"))
        stats.update()
    return


@hydra.main(version_base=None, config_path="../../conf/experiments")
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
    dl_lib_type = cfg["training"]["type"]

    train = dataset.get_train_dataset()
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )

    if not skip_test:
        test = dataset.get_test_dataset()
        test_loader = DataLoader(
            dataset=test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            prefetch_factor=2,
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

    mlflow_logger.log_artifact(get_logger_path("train.log"))

    torch.cuda.empty_cache()
    if dl_lib_type == "SLAYER":
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
    elif dl_lib_type == "Bootstrap":
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
