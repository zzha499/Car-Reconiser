import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Foreshadow (yzhb120, zzha499)", # Replace with your own username
    version="0.0.1",
    author="Will Zhang, Eric Zhao",
    author_email="zzha499@aucklanduni.ac.nz, @aucklanduni.ac.nz ",
    description="Car Recogniser",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UOA-CS302-2020/CS302-Python-2020-Group5",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)