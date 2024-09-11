# EVDAD

An EVent-Driven Anomaly Detection for Spiking Neural Networks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.10
- CUDA 12.3 or 12.5

### Installing

A step by step series of examples that tell you how to get a development env running.

#### [Linux]

Clone the repo and create virtual environment.

```bash
cd $HOME
git clone git@github.com:Piotr45/evdad.git evdad
cd evdad
python -m venv evdad-venv
source evdad-venv/bin/activate
```

Install requirements.

```bash
pip install -r requirements.txt
```

Install EVDAD package.

```bash
pip install -e .
hash -r
```

## Running the tests

TODO if time

## Tutorials

- [End to end training tutorials](docs/tutorials/training/README.md)
- [Inference Tutorial](docs/tutorials/training/README.md)

## Built With

* [LAVA](https://lava-nc.org/index.html) - The SNN training framework
* [Hydra](https://hydra.cc) - Used for configs
* [MLflow](https://mlflow.org) - Used to track training

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

- **Piotr Baryczkowski** - *Initial work, framework maintenance* - [Piotr45](https://github.com/Piotr45)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

- Hat tip to creators of [V2E](https://sites.google.com/view/video2events/home) and [LAVA](https://lava-nc.org/index.html).