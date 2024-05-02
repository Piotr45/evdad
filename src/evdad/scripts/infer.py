import logging

import cv2
import hydra
import numpy as np
import setproctitle
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from evdad.utils import get_device

log = logging.getLogger(__name__)


def events_to_image(events: np.ndarray) -> np.ndarray:
    on_events = np.sum(np.transpose(events, (0, 2, 1, 3)), axis=0)[:, :, 0]
    on_events[on_events >= 1] = 255
    on_events[on_events == 0] = 128
    off_events = np.sum(np.transpose(events, (0, 2, 1, 3)), axis=0)[:, :, 1]
    off_events[off_events >= 1] = -1
    off_events[off_events == 0] = 128
    off_events[off_events == -1] = 0
    image = on_events + off_events
    return image.astype(np.int8)


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    setproctitle.setproctitle("evdad-infer")

    assert cfg["model"]

    log.info(OmegaConf.to_yaml(cfg))

    device = get_device()
    log.info(f"Using device: {device}")

    net = hydra.utils.instantiate(cfg["model"]).to(device)
    state = torch.load(cfg["processing"]["checkpoint"])
    net.load_state_dict({"checkpoint": state}, strict=False)

    dataset = hydra.utils.instantiate(cfg["dataset"])

    train = dataset.get_train_dataset()
    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=False)

    shape = [net.C, net.H, net.W, net.T]

    for idx, (im, label) in enumerate(train_loader):
        reconstructed = net(im.to(device))

        reconstructed = np.reshape(reconstructed[0].cpu().detach().numpy(), shape).T
        img = np.reshape(im[0].cpu().detach().numpy(), shape).T

        out = np.hstack((events_to_image(img), events_to_image(reconstructed)))
        cv2.imshow("Infer", out)
        cv2.waitKey(0)
        if idx == 5:
            return
    return


if __name__ == "__main__":
    main()
