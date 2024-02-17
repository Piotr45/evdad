import hydra
import lava.lib.dl.slayer as slayer
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    cfg = cfg["defaults"]

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
        count_log=True,
    )

    for epoch in range(epochs):
        print(epoch)
        for i, (input, label) in enumerate(train_loader):  # training loop
            print(i)
            output, _ = assistant.train(input, label)
            print(
                f"\r[Epoch {epoch:2d}/{epochs}][Progress {i*batch_size}/{len(train)}] {stats}",
                end="",
            )

        for i, (input, label) in enumerate(test_loader):  # test loop
            output, _ = assistant.test(input, label)
            print(
                f"\r[Epoch {epoch:2d}/{epochs}][Progress {i*batch_size}/{len(test)}] {stats}",
                end="",
            )

        stats.update()


if __name__ == "__main__":
    main()
