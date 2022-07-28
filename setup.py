import setuptools

required = ["numpy", "matplotlib", "scipy", "tqdm", "jupyter"]

setuptools.setup(
    name="ratinabox",
    version="0.1",
    description="RatInABox: A package for simulating motion and ephys data in continuous environments",
    author="Tom George",
    author_email="tomgeorge1@btinternet.com",
    long_description_content_type="text/markdown",
    url="https://github.com/TomGeorge1234/RatInABox",
    download_url="https://github.com/TomGeorge1234/RatInABox",
    packages=setuptools.find_packages(),
    install_requires=required,
    license="Apache License 2.0",
)

