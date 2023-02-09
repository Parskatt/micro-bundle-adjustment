from setuptools import setup

setup(
    name = "micro-bundle-adjustment",
    packages=["micro_bundle_adjustment"],
    version="0.0.1",
    author="Johan Edstedt",
    install_requires=open("requirements.txt", "r").read().split("\n"),
)
