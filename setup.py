from setuptools import setup

setup(
    name="evdad",
    version="0.1.0",
    description="My masters thesis project",
    package_dir={"": "src"},
    package_data={"": ["*.yaml", "**/*.yaml"]},
    include_package_data=True,
)
