from setuptools import setup, find_packages
import os

# Read the contents of the README.md file for the long description
with open("README.MD", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ovrolwathings",  # Replace with the name of your package
    version="0.1.0",  # Initial version; update as needed",
    description="A package for OVRO-LWA radio astronomy data processing and analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peijin94/ovro-lwa-things",
    packages=find_packages(),  # Automatically find packages in the repository
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update if using a different license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Adjust as per the codebase compatibility
    install_requires=[],  # Install dependencies from requirements.txt
)
