import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Foreshadow (yzhb120, zzha499)", # Replace with your own username
    version="0.0.1",
    author="Will Zhang, Eric Zhao",
    author_email="zzha499@aucklanduni.ac.nz, @aucklanduni.ac.nz ",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)