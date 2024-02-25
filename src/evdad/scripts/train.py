import hydra
import lava.lib.dl.slayer as slayer
import mlflow
import torch
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    cfg = cfg["defaults"]

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO better config reading
    dataset = hydra.utils.instantiate(cfg["dataset"])
    batch_size = cfg["dataloader"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["optimizer"]["lr"]

    train = dataset.get_train_dataset()
    test = dataset.get_test_dataset()

    net = hydra.utils.instantiate(cfg["model"]).to(device)

    # TODO setup functions for optimizer, loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_loader = DataLoader(
        dataset=train, batch_size=batch_size, shuffle=True
    )  # , num_workers=2, prefetch_factor=2)
    test_loader = DataLoader(
        dataset=test, batch_size=batch_size, shuffle=True
    )  # , num_workers=2, prefetch_factor=2)

    error = slayer.loss.SpikeRate(true_rate=0.3, false_rate=0.05, reduction="sum").to(
        device
    )

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net,
        error,
        optimizer,
        stats,
        classifier=slayer.classifier.Rate.predict,
    )

    mlflow_logger = hydra.utils.instantiate(cfg["mlflow"])

    mlflow_logger.run_experiment()

    mlflow_logger.log_param("epochs", epochs)
    mlflow_logger.log_param("batch size", batch_size)
    mlflow_logger.log_param("learning rate", lr)

    torch.cuda.empty_cache()
    for epoch in range(epochs):
        for i, (input, label) in enumerate(train_loader):  # training loop
            output = assistant.train(input.to(device, dtype=torch.float), label)
            stats.print(epoch, iter=i, dataloader=train_loader)

        mlflow_logger.log_metric("training_loss", stats.training.loss, step=epoch)
        mlflow_logger.log_metric(
            "training_accuracy", stats.training.accuracy, step=epoch
        )

        for i, (input, label) in enumerate(test_loader):  # test loop
            output = assistant.test(input.to(device, dtype=torch.float), label)
            stats.print(epoch, iter=i, dataloader=test_loader)

        mlflow_logger.log_metric("test_loss", stats.testing.loss, step=epoch)
        mlflow_logger.log_metric("test_accuracy", stats.testing.accuracy, step=epoch)

        stats.update()

    mlflow.pytorch.log_model(net, mlflow_logger.run_id)
    mlflow_logger.finish_experiment()
    print("Model training and logging complete.")


if __name__ == "__main__":
    main()
