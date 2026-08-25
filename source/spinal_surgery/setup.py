"""Installation script for the 'spinal_surgery' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "matplotlib",
    "nibabel",
    "opencv-python",
    "psutil",
    "pyvista",
    "ruamel.yaml",
    "scipy",
    "sonogym-reconstruction-data==0.1.2",
]


def collect_package_data():
    package_root = os.path.join(EXTENSION_PATH, "spinal_surgery")
    data_files = []
    for root, dirs, files in os.walk(package_root):
        rel_root = os.path.relpath(root, package_root)
        rel_parts = set(rel_root.split(os.sep))
        if "__pycache__" in rel_parts:
            continue
        if rel_root.startswith(os.path.join("assets", "data", "HumanModels")):
            dirs[:] = []
            continue
        for filename in files:
            if filename.endswith((".py", ".pyc", ".pyo")):
                continue
            rel_path = os.path.normpath(os.path.join(rel_root, filename))
            if rel_path == ".":
                rel_path = filename
            data_files.append(rel_path)
    return data_files

# Installation operation
setup(
    name="sonogym-reconstruction-core",
    packages=find_packages(),
    py_modules=["sonogym_bootstrap"],
    author="SonoGym Reconstruction Contributors",
    maintainer="SonoGym Reconstruction Contributors",
    url="https://github.com/isaac-sim/IsaacLabExtensionTemplate.git",
    version=EXTENSION_TOML_DATA["package"]["version"],
    description="Core SonoGym robotic ultrasound reconstruction task, sensors, controllers, and assets.",
    keywords=["sonogym", "robotic ultrasound", "isaaclab", "reconstruction"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    package_data={"spinal_surgery": collect_package_data()},
    entry_points={
        "console_scripts": [
            "sonogym-data=spinal_surgery.tools.data:main",
            "sonogym-bootstrap=sonogym_bootstrap:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.0.0",
    ],
    zip_safe=False,
)
