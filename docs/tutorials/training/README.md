# EVDAD Training

In this tutorial you will learn how to use `evdad-train` script, for neural network training.

## Running the training

To run evdad-train you will need to launch MLflow app. To do this, create new tmux session or run this command in other terminal.

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

**NOTE:** This will create two folders *mlruns* and *mlartifacts*. It is prefered to this in separate directory or in *evdad* main directory.

Then in other terminal, we will use `evdad-bootstrap-train` or `evdad-slayer-train` console script to run the training.

```bash
evdad-bootstrap-train --config-name=experiment_name hydra.run.dir=/path/to/output/dir
evdad-slayer-train --config-name=experiment_name hydra.run.dir=/path/to/output/dir     
```

Flags:

 - `config-name` - specifies experiment name (from *src/conf* directory e.g.experiment_NMNIST_cuba_cnn_autoencoder.yaml)
 - `hydra-run-dir` - the hydra output directory (where we want to store training data). *Defaults* to *outputs/DATE/HOUR* dir in your current path.

 ### Example config

 ```yaml
defaults:
  - mlflow: local
  - train
  - dataset: HTVD
  - model: slayer_sdnn_cnn_autoencoder
  - _self_
  
mlflow:
  experiment_name: SLAYER
  run_name: slayer_sdnn_NMNIST_cnn_autoencoder
  tags: {version: 0.1.0}

loss:
  loss_function: SpikeTime

hydra:
  job:
    chdir: true
  searchpath:
    - pkg://conf.defaults
 ``` 

Example command

```bash
evdad-slayer-train --config-name=slayer_train_autoencoder_CUBA_dense hydra.run.dir=$SNN_EXPERIMENTS/$(date +%Y-%m-%d_%H-%M-%S)_piotr_debug training.epochs=1 +dataset.use_cache=true 
```