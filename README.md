# master-thesis
## Using this as a Python package
This project uses pyo3 to expose the basic structures (e.g. `Problem`, `Solution`) in the Rust implementation as a python package. It uses [Maturin](https://github.com/PyO3/maturin) for building python wheels from source. Installing this project as a package in the active python environment should be as simple as doing:
```shell
pip install git+https://github.com/fiskeforgutane/master-thesis.git
```

It should also be possible to add `git+https://github.com/fiskeforgutane/master-thesis.git` into a `requirements.txt` file.

## Problem status for `t60`

| Problem                 | Solved | Note |
| ----------------------- | ------ | ---- |
| LR1_1_DR1_3_VC1_V7a     |   ✔️   |      |
| LR1_1_DR1_4_VC3_V12b    |   ✔️   |      |
| LR1_2_DR1_3_VC2_V6a     |   ✔️   | Solved for `t120`, should be good |
| LR2_11_DR2_33_VC4_V11a  |   ✔️   |      |
| LR2_22_DR3_333_VC4_V14a |   ✔️   |      |
| LR1_1_DR1_4_VC3_V11a    |   ✔️   |      |
| LR1_1_DR1_4_VC3_V8a     |   ✔️   |      |
| LR1_2_DR1_3_VC3_V8a     |   ✔️   |      |
| LR2_11_DR2_33_VC5_V12a  |   ✔️   | using rolling horizon stepsize 3     |
| LR2_22_DR3_333_VC4_V17a |   ❗   |      |
| LR1_1_DR1_4_VC3_V12a    |   ✔️   |      |
| LR1_1_DR1_4_VC3_V9a     |   ✔️   |      |
| LR2_11_DR2_22_VC3_V6a   |   ✔️   |      |
| LR2_22_DR2_22_VC3_V10a  |   ✔️   |      |
