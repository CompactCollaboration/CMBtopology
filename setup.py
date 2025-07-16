from setuptools import setup, find_packages

setup(
    name="cmb-topology",
    version="0.1.0",
    description="A Python library for computing CMB covariance matrices for non-trivial topologies.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Amirhossein Samandar",
    author_email="amirsamandar@gmail.com",
    url="https://github.com/[your-username]/cmb-topology-covariance",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        "camb>=1.3.0",
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "numba>=0.53",
        "tqdm>=4.60"
        "spherical>=1.0.14",
        "quaternionic>=1.0.13"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    python_requires=">=3.12"
)