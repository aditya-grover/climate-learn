import setuptools

setuptools.setup(
    name = "climate_tutorial",
    version = "0.0.1",
    author = "Hritik Bansal, Shashank Goel, Tung Nguyen, Aditya Grover",
    author_email = "hbansal@ucla.edu, shashankgoel@ucla.edu, tungnd@ucla.edu, agrover@ucla.edu",
    description = "Climate Tutorial",
    long_description = open("README.md", "r").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/tung-nd/climate_tutorial",
    packages = setuptools.find_packages(),
    install_requires = [
        "rich",
        "timm",
        "cdsapi",
        "pytorch-lightning"
    ],
    classifiers = [
        "Development Status :: In Progress"
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)