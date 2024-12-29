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

In the *data* directories we will store *.bin* or *.npy* files with the event driven video data.

In the *labels* dir we will store labels for the whole sequence or for each frame in  the *.bin* (or *.npy*) sample with the same name.

Other files:

- *test.csv* - paths to test data
- *test_labels.csv* - paths to test labels
- *train.csv* - paths to train data
- *train_labels.csv* - paths to train labels

## NMNIST

The [Neuromorphic-MNIST](https://www.garrickorchard.com/datasets/n-mnist) (N-MNIST) dataset is a spiking version of the original frame-based MNIST dataset. It consists of the same 60 000 training and 10 000 testing samples as the original MNIST dataset, and is captured at the same visual scale as the original MNIST dataset (28x28 pixels). The N-MNIST dataset was captured by mounting the ATIS sensor on a motorized pan-tilt unit and having the sensor move while it views MNIST examples on an LCD monitor as shown in this video. A full description of the dataset and how it was created can be found in the paper below. 

- Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.  “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades", Frontiers in Neuroscience, vol.9, no.437, Oct. 2015 ([open access Frontiers link](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2015.00437/full))

**NOTE**: In case of NMNIST dataset you dont need csv files, and the database should look like this:

```
dataset
   ├── test
   │   ├── 0
   │   │   └─ 00002.bin
   │   └── 1
   │       └─ 00004.bin
   └── train
       ├── 0
       │   └─ 00002.bin
       └── 1
           └─ 00004.bin
```

![nmnist example](/docs/assets/nmnist_data_example.JPG)

## HTVD Highway Traffic Videos Dataset

This a [database](https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset) of video of traffic on the highway used in [1] and [2]. The video was taken over two days from a stationary camera overlooking I-5 in
Seattle, WA. The video were labeled manually as light, medium, and heavy
traffic, which correspond respectively to free-flowing traffic, traffic at
reduced speed, and stopped or very slow speed traffic. The training and test
sets used in [1] and [2] are also provided. The first frame of the original
video is corrupted with another video signal, so when processing each video
you will have to start from the 2nd frame. In addition, the cropped versions
of the video used in [1,2] are provided in a MATLAB file. The video is
provided courtesy of Washington State Department of Transportation
[http://www.wsdot.wa.gov/].

**NOTE**: This dataset was converted to Event Driven Videos with [Easy V2E](https://github.com/Piotr45/easy-v2e).

![data comparison](/docs/assets/htvd_data_comparison.png)

## UBI-Fights

**NOTE: Currently not supproted**

## UCSD Anomaly Detection Dataset

The [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) was acquired with a stationary camera mounted at an elevation, overlooking pedestrian walkways. The crowd density in the walkways was variable, ranging from sparse to very crowded. In the normal setting, the video contains only pedestrians. Abnormal events are due to either:

the circulation of non pedestrian entities in the walkways
anomalous pedestrian motion patterns
Commonly occurring anomalies include bikers, skaters, small carts, and people walking across a walkway or in the grass that surrounds it. A few instances of people in wheelchair were also recorded. All abnormalities are naturally occurring, i.e. they were not staged for the purposes of assembling the dataset. The data was split into 2 subsets, each corresponding to a different scene. The video footage recorded from each scene was split into various clips of around 200 frames.

Peds1: clips of groups of people walking towards and away from the camera, and some amount of perspective distortion. Contains 34 training video samples and 36 testing video samples.

Peds2: scenes with pedestrian movement parallel to the camera plane. Contains 16 training video samples and 12 testing video samples.

For each clip, the ground truth annotation includes a binary flag per frame, indicating whether an anomaly is present at that frame. In addition, a subset of 10 clips for Peds1 and 12 clips for Peds2 are provided with manually generated pixel-level binary masks, which identify the regions containing anomalies. This is intended to enable the evaluation of performance with respect to ability of algorithms to localize anomalies.

**NOTE**: This dataset was converted to Event Driven Videos with [Easy V2E](https://github.com/Piotr45/easy-v2e).

## References

[1] A. B. Chan and N. Vasconcelos, "Probabilistic Kernels for the Classification
of Auto-Regressive Visual Processes". Proceedings of IEEE Conference on
Computer Vision and Pattern Recognition, San Diego, 2005.

[2] A. B. Chan and N. Vasconcelos, "Classification and Retrieval of Traffic
Video using Auto-Regressive Stochastic Processes". Proceedings of 2005
IEEE Intelligent Vehicles Symposium, Las Vegas, June 2005.