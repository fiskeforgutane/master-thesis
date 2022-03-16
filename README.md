# master-thesis
## Using this as a Python package
This project uses pyo3 to expose the basic structures (e.g. `Problem`, `Solution`) in the Rust implementation as a python package. It uses [Maturin](https://github.com/PyO3/maturin) for building python wheels from source. Installing this project as a package in the active python environment should be as simple as doing:
```shell
pip install maturin
pip install git+https://github.com/fiskeforgutane/master-thesis.git
```

It should also be possible to add `git+https://github.com/fiskeforgutane/master-thesis.git` into a `requirements.txt` file.
