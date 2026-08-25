"""Installation script for the SonoGym reconstruction data locator package."""

from __future__ import annotations

from setuptools import find_packages, setup


setup(
    name="sonogym-reconstruction-data",
    version="0.1.1",
    description="SonoGym reconstruction data locator and automatic downloader.",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
