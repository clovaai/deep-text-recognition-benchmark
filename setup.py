from setuptools import setup, find_packages


def install_deps():
    """Reads requirements.txt and preprocess it
    to be feed into setuptools.

    This is the only possible way (we found)
    how requirements.txt can be reused in setup.py
    using dependencies from private github repositories.

    Links must be appended by `-{StringWithAtLeastOneNumber}`
    or something like that, so e.g. `-9231` works as well as
    `1.1.0`. This is ignored by the setuptools, but has to be there.

    Warnings:
        to make pip respect the links, you have to use
        `--process-dependency-links` switch. So e.g.:
        `pip install --process-dependency-links {git-url}`

    Returns:
         list of packages and dependency links.
    """
    pkgs = []
    links = []

    with open("requirements.txt", "r") as req_file:
        for resource in req_file.readlines():
            if "git+ssh" in resource:
                links.append(resource.strip())
            else:
                pkgs.append(resource.strip())
    return pkgs, links


new_pkgs, new_links = install_deps()

setup(
    name="trba",
    package_dir={"": "src"},
    packages=find_packages("src", exclude=["test", "test.*"]),
    version="v0.1.1",
    install_requires=new_pkgs,
    dependency_links=new_links,
)


# setup(

#     name ="dptr", 
#     version = '0.1',    
#     packages = find_packages(),     
#     install_requires = ['numpy', 'lmdb', 'natsort', 'Pillow', 'six', 'torch', 'torchvision','typing-extensions'], 
    
    
#     description = "A package for TRBA model for OCR",
# )
