# Documentation
The documentation source files are located in the `docs/source` folder and should be written in either [reStructuredText](https://docutils.sourceforge.io/rst.html) or [markdown](https://myst-parser.readthedocs.io/en/stable/syntax/typography.html). The `index.rst` file corresponds to the main page of the documentation website. Other `.rst`  or `.md` files can be included in the main page via the `toctree` directive.

The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/) and the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/latest/). The `docs/source/conf.py` file contains the `Sphinx` configuration.

## Building the documentation locally
You can build and view the documentation website locally, on your machine. To do so, run the following commands from the root of your project:
```sh
# Install the documentation build dependencies
pip install -r docs/requirements.txt
# Build the documentation
sphinx-build docs/source docs/build
```
This should create a `docs/build` folder. You can view the local build by opening `docs/build/index.html` in a browser.
To refresh the documentation, after making changes, remove the `docs/build` folder and re-run the above command:

```sh
rm -rf docs/build
sphinx-build docs/source docs/build
```

## Publishing the documentation
We have included an extra GitHub actions workflow in `.github/workflows/docs_build_and_deploy.yml` that will build the documentation and deploy it to [GitHub pages](https://pages.github.com/).
* The build step is triggered every time a pull request is opened or a push is made to the `main` branch. This way you can make sure that the documentation does not break before merging your changes.
* The deployment is triggered only when a tag is present. This ensures that new documentation versions are published in tandem with the release of a new package version on PyPI.
* The published docs are by default hosted at `https://<github_username_or_organization>.github.io/<package_name>/`. To enable hosting, you will need to go to the settings of your repository, and under the "Pages" section, select the `gh-pages` branch as the source for your GitHub pages site.
* A popular alternative to GitHub pages for hosting the documentation is [Read the Docs](https://readthedocs.org/). To enable hosting on Read the Docs, you will need to create an account on the website and follow the instructions to link your GitHub repository to your Read the Docs account.

