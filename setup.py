import setuptools

with open(".README_PyPI.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="ratinabox",
    version="1.0.6",
    description="RatInABox: A package for simulating motion and ephys data in continuous environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tom George",
    author_email="tomgeorge1@btinternet.com",
    url="https://github.com/TomGeorge1234/RatInABox",
    download_url="https://github.com/TomGeorge1234/RatInABox",
    packages=setuptools.find_packages(),
    package_data={"ratinabox": ["data/*"]},
    python_requires=">=3.7",
    install_requires=required,
    license="MIT License",
)
