import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ratinabox",
    version="0.1",
    description="RatInABox: A package for simulating motion and ephys data in continuous environments",
    author="Tom George",
    author_email="tomgeorge1@btinternet.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomGeorge1234/RatInABox",
    download_url="https://github.com/TomGeorge1234/RatInABox",
    packages=setuptools.find_packages(),
    license="Apache License 2.0",
)

