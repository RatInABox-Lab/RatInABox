import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ratinabox",
    version="0.1",
    author="Tom George",
    author_email="tom.george.20@ucl.ac.uk",
    description="A package for simulation motion and ephys data in continuous environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomGeorge1234/RatInABox",
    packages=setuptools.find_packages(),
)





