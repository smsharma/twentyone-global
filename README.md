# twentyone-global

Simplified framework for modeling the global 21-cm absorption signal, with a focus on modeling non-standard 21-cm CMB temperature evolution.

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

![21-cm absorption temperature for various CMB photon temperature evolutions.](notebooks/plots/banner.png)

Derived 21-cm absorption temperature (left) for various amounts of excess power-law photon temperature parameterization (right) from [Fialkov and Barkana (2019)](https://arxiv.org/abs/1902.02438).

## Code

The dependencies of the code are listed in [environments.yml](environment.yml). In addition to the usual Python scientific stack, the `astropy`, `classy`, `colossus`, and `tqdm` packages are required.

## Notebooks

The notebook [01_basic_examples.ipynb](notebooks/01_basic_examples.ipynb) contains basic high-level examples for running the code.

## Authors

-  Andrea Caputo; andrea dot caputo at uv dot es
-  Hongwan Liu; hongwanl at princeton dot edu
-  Siddharth Mishra-Sharma; sm8383 at nyu dot edu
-  Joshua T. Ruderman; ruderman at nyu dot edu

## References

Various components of the code are based on [astro-ph/0507102](https://arxiv.org/abs/astro-ph/0507102), [1804.02406](https://arxiv.org/abs/1804.02406), [0910.4410](https://arxiv.org/abs/0910.4410), [0802.2102](https://arxiv.org/abs/0802.2102), [1312.4948](https://arxiv.org/abs/1312.4948), and [1904.09296](https://arxiv.org/abs/1904.09296).