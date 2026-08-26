ALADIN: Accuracy–Latency–Aware Design-space Inference Analysis \\ for Embedded AI Accelerators
===================================
ALADIN is a HW--SW co-design evaluation tool built on top of the DORY deployment
framework. The tool guides the design-space exploration of quantized neural
networks, from the selection of quantization and implementation strategies
to their evaluation on the target platform.

**Note:** The artifact is anonymized because the associated paper is
currently under review. Parts of the codebase are derived from the
open-source DORY project and therefore retain the corresponding project
structure and references.

Installation
------------

1. Clone the repository and the required submodules.
2. Build the Dockerfile with the required SDKs and Python env
```
cd ALADIN
docker buildx build -t dory-docker:3.9 ./.devcontainer/ 
```

3. Once connected to the terminal of the container run the following commands:
```
source /dory_env/bin/activate
source docker_util/docker_pulp_sdk.sh
```
*NOTE: these packages cannot be installed from the Dockerfile*

Experimets
---------
The main implementation of ALADIN is located in: `./dory/Frontend_frameworks/QONNX`.

The experiments presented in the paper can be reproduced using the Jupyter
notebook: `./notebooks/aladin.ipynb`.

The notebook contains the configuration and execution of the experiments
used to evaluate the proposed work.


### Reference
*We are NOT the developers of DORY project*, however if you are interested please consider to cite also their paper: https://ieeexplore.ieee.org/document/9381618 (preprint available also at https://arxiv.org/abs/2008.07127)
```
@article{burrello2020dory,
  author={A. {Burrello} and A. {Garofalo} and N. {Bruschi} and G. {Tagliavini} and D. {Rossi} and F. {Conti}},
  journal={IEEE Transactions on Computers}, 
  title={DORY: Automatic End-to-End Deployment of Real-World DNNs on Low-Cost IoT MCUs}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TC.2021.3066883}
}
```

### Contributors
+ **Tommaso Baldi**, *SSSA*, [email](mailto:tommaso.baldi@santannapisa.it)


### License
This project and DORY are released under Apache 2.0, see the LICENSE file in the root of this repository for details.
