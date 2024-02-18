import hydra
import lava.lib.dl.slayer as slayer
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    cfg = cfg["defaults"]

    print(OmegaConf.to_yaml(cfg))

    # device = torch.device("cpu")
    device = torch.device("cuda")

    # TODO better config reading
    dataset = hydra.utils.instantiate(cfg["dataset"])
    batch_size = cfg["dataloader"]["batch_size"]
    epochs = cfg["training"]["epochs"]

    train = dataset.get_train_dataset()
    test = dataset.get_test_dataset()

    net = hydra.utils.instantiate(cfg["model"]).to(device)

    # TODO setup functions for optimizer, loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

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

    mlflow.set_experiment("LAVA_training")

    # Start an MLflow run
    with mlflow.start_run() as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", 0.001)
        for epoch in range(epochs):
            for i, (input, label) in enumerate(train_loader):  # training loop
                output = assistant.train(input, label)
                stats.print(epoch, iter=i, dataloader=train_loader)

            for i, (input, label) in enumerate(test_loader):  # test loop
                output = assistant.test(input, label)
                stats.print(epoch, iter=i, dataloader=test_loader)

            mlflow.log_metric("taining_loss", stats.training.loss, step=epoch)
            mlflow.log_metric("taining_accuracy", stats.training.accuracy, step=epoch)
            mlflow.log_metric("test_loss", stats.testing.loss, step=epoch)
            mlflow.log_metric("test_accuracy", stats.testing.accuracy, step=epoch)
            stats.update()

    print("Model training and logging complete.")


if __name__ == "__main__":
    main()
