import os
import setuptools
import pkg_resources

setuptools.setup(
    name = "climate_learn",
    py_modules = ["climate_learn"],
    version = "0.0.1",
    author = "Hritik Bansal, Shashank Goel, Tung Nguyen, Aditya Grover",
    author_email = "hbansal@ucla.edu, shashankgoel@ucla.edu, tungnd@ucla.edu, agrover@ucla.edu",
    description = "Climate Learn",
    long_description = open("README.md", "r").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/aditya-grover/climate-learn",
    packages = setuptools.find_packages(),
    install_requires = [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements", "main.txt"))
        )
    ],
    extras_require = {
        "docs": [
            str(r)
            for r in pkg_resources.parse_requirements(
                open(os.path.join(os.path.dirname(__file__), "requirements", "docs.txt"))
            )
        ]
    },
    classifiers = [
        "Development Status :: In Progress"
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
)