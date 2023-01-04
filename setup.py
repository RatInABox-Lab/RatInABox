import setuptools

long_description = """
`RatInABox` (paper [here](https://www.biorxiv.org/content/10.1101/2022.08.10.503541v1)) is a toolkit for generating locomotion trajectories and complementary neural data for spatially and/or velocity selective cell types. With `RatInABox` you can: 

* **Generate realistic trajectories** for rats exploring complex 1- and 2-dimensional environments under a smooth random policy, an external control signal, or your own trajectory data.
* **Generate artificial neuronal data** Simulate various location or velocity selective cells found in the Hippocampal-Entorhinal system, or build your own more complex cell type. 
* **Build and train complex networks** Build, train and analyse complex networks of cells, powered by data generated with `RatInABox`. 

For full details, tutorials, source code and more please see our [github repository](https://github.com/TomGeorge1234/RatInABox)
"""

setuptools.setup(
    name="ratinabox",
    version="1.0.0",
    description="RatInABox: A package for simulating motion and ephys data in continuous environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tom George",
    author_email="tomgeorge1@btinternet.com",
    url="https://github.com/TomGeorge1234/RatInABox",
    download_url="https://github.com/TomGeorge1234/RatInABox",
    packages=setuptools.find_packages(),
    package_data={
        "":["requirements.txt"],
        "ratinabox": ["data/*"]},
    python_requires=">=3.7",
    install_requires=[
        'numpy~=1.23.3',
        'matplotlib~=3.5.3',
        'scipy~=1.9.3',
    ],
    license="MIT License",
)
