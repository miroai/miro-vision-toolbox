# miro-vision-toolbox

## Examples

## Installing

## Development

### Building from Source
To install and test the `miro-vision-toolbox` locally:
```
git clone --shallow
cd miro-vision-toolbox
pip install -e .
```

### Tests
_Before pushing your commits, please make sure tests are were!_ Tests are included in each function's docstring and can be ran using Python's [`doctest`](https://realpython.com/python-project-documentation-with-mkdocs/#write-examples-and-test-them-using-doctest). To run tests on `toolbox.img_utils`:

```
python -m doctest -v toolbox/img_utils.py
```
To learn more about `doctest` see [here](https://realpython.com/python-doctest/)

### Documentation Generation
Docs can be automagically generated using `mkdocs` (see [RealPython's excellent tutorial here](https://realpython.com/python-project-documentation-with-mkdocs/#step-1-set-up-your-environment-for-building-documentation)). Additionally examples are generated using Jupyter notebook and converted into documentation thanks to [`mkdocs-jupyter`](https://github.com/danielfrg/mkdocs-jupyter).
This package's documentation is located [here (github page)](https://miroai.github.io/miro-vision-toolbox/).

Running `mkdocs gh-deploy` will update the live site (to be ran only after testing). To see the documentation page locally before pushing, run `mkdocs serve` then go to `http://localhost:800`

### Resource
* this package was build with inspiration from [this](https://www.freecodecamp.org/news/build-your-first-python-package/) and [this post](https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/)
