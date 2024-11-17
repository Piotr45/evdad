from setuptools import setup

setup(
    name="evdad",
    version="0.2.0",
    author="Piotr45",
    author_email="piotr.baryczkowski@put.poznan.pl",
    description="My masters thesis project",
    package_dir={"": "src"},
    package_data={"": ["*.yaml", "**/*.yaml", "conf"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "evdad-train = evdad.scripts.train:main",
            "evdad-infer = evdad.scripts.infer:main",
        ],
    },
)
