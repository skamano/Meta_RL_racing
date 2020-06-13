# ECE276B_PR4
Final Project for ECE276B

# Prepare environment
```bash
conda env create -f environment.yml
conda activate project4
cd garage
pip install .
```

Some packages may be incompatible and cause strange errors. If this happens to you, use pip to force the correct package
versions as they show up in your installation prop.
```bash
pip uninstall pyglet cloudpickle
pip install pyglet==1.3.2 cloudpickle==1.2.0
```

Environments given by [AirSim NeurIPS 2019 Drone Racing API](https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing)

[This implementation of Meta-RL](https://github.com/cbfinn/maml_rl) uses rllab, which is now maintained under the name
 [garage](https://github.com/rlworkgroup/garage) which provides PyTorch+TensorFlow support for reinforcement learning algorithms.
While I did not manage to formulate enough tasks and create a training routine for racing across many environments, I set
up an OpenAI-gym environment which interfaces with the `garage` and `airsimneurips` frameworks and implemented a basic
policy gradient algorithm (vanilla policy gradient, or VPG) as as starting point.
