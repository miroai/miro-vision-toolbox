# miro-vision-toolbox

## Usage

## Installing


```
git clone --shallow
cd miro-vision-toolbox
pip install -e .
```

## Deployment (dev only)
### Tests
tests are included in each function's docstring and can be ran using Python's [`doctest`](https://realpython.com/python-project-documentation-with-mkdocs/#write-examples-and-test-them-using-doctest). To run tests on `toolbox.img_utils`:

```
python -m doctest -v toolbox/img_utils.py
```
To learn more about `doctest` see [here](https://realpython.com/python-doctest/)
### Documentation Generation
Docs should be available [here (github page)](https://miroai.github.io/miro-vision-toolbox/) after running this command:
```
mkdocs gh-deploy
```

### Resource
* this package was build with inspiration from [this](https://www.freecodecamp.org/news/build-your-first-python-package/) and [this post](https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/)
* the documentation page is setup using `mkdocs` following [this tutorial](https://realpython.com/python-project-documentation-with-mkdocs/#step-1-set-up-your-environment-for-building-documentation)
