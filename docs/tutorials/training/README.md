# EVDAD Training

In this tutorial you will learn how to use `evdad-train` script, for neural network training.

## Running the training

To run evdad-train you will need to launch MLflow app. To do this, create new tmux session or run this command in other terminal.

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

**NOTE:** This will create two folders *mlruns* and *mlartifacts*. It is prefered to this in separate directory or in *evdad* main directory.

Then in other terminal, we will use `evdad-train` console script to run the training.

```bash
evdad-train --config-name=experiment_name hydra.run.dir=/path/to/output/dir     
```

Flags:

 - `config-name` - specifies experiment name (from *src/conf* directory e.g.experiment_NMNIST_cuba_cnn_autoencoder.yaml)
 - `hydra-run-dir` - the hydra output directory (where we want to store training data). *Defaults* to *outputs/DATE/HOUR* dir in your current path.

 ### Example config

 ```yaml
 defaults:
  - mlflow: local
  - train
  - dataset: NMNIST
  - model: slayer_cuba_cnn_autoencoder
  - _self_
  
mlflow:
  experiment_name: SLAYER
  run_name: slayer_cuba_NMNIST_cnn_autoencoder
  tags: {version: 0.1.0}

training:
  epochs: 5

dataset:
  data_is_label: true

dataloader:
  batch_size: 128

loss:
  loss_function: SpikeTime
  time_constant: 5
  length: 100
  filter_order: 1
  reduction: sum

hydra:
  job:
    chdir: true
  searchpath:
    - file:///${oc.env:HOME}/evdad/src/conf/defaults
 ``` 