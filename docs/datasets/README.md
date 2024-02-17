# Datasets

Datasets used in project:

1. [Highway Traffic Videos Dataset](https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset) (HTVD)
2. [UBI-Fights](https://paperswithcode.com/dataset/ubi-fights) (Abnormal Event Detection Dataset)
3. [UCSD](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) Anomaly Detection Dataset

## Dataset structure

The file structure below is the same for each dataset used in this project.

```
dataset
   ├── test
   │   ├── data
   │   │   └─ cctv052x2004080516x01638.bin
   │   └── labels
   │       └─ cctv052x2004080516x01638.csv
   ├── train
   │   ├── data
   │   │   └─ cctv052x2004080517x01663.bin
   │   └── labels
   │       └─ cctv052x2004080517x01663.csv
   ├── test.csv
   ├── test_labels.csv
   ├── train.csv
   └── train_labels.csv
```

In the *data* directories we will store *.bin* files with the event driven video data.

In the *labels* dir we will store labels for the whole sequence or for each frame in  the *.bin* sample with the same name.


## HTVD

TODO

## UBI-Fights

TODO

## UCSD

TODO