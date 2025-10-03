Mixed-precision QNN co-design tool for MCU
===================================

DORY is an automatic tool to deploy DNNs on low-cost MCUs with typically less than 1MB of on-chip SRAM memory. 


Installation
------------

1. Clone the repository and the required submodules.
2. Build the Dockerfile with the required SDKs and Python env
```
cd mixed-precision-dory
docker build -t dory-docker:3.9 ./.devcontainer/ 
```

Experimets
---------
todo ...


Examples
--------
To download the examples built on DORY, clone the internal dory_example submodule (it should be automatically previously downloaded).
Then, you can run one example from the library with the following command:
```
python3 network_generate.py NEMO PULP.PULP_gvsoc ./dory/dory_examples/config_files/config_NEMO_MV1.json --app_dir ./application/
```


### Reference
Project build on top of DORY tool, please make sure to cite also their paper: https://ieeexplore.ieee.org/document/9381618 (preprint available also at https://arxiv.org/abs/2008.07127)
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
Project and DORY are released under Apache 2.0, see the LICENSE file in the root of this repository for details.
