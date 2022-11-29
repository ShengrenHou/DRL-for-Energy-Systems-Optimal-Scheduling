
# Performance Comparison of Deep RL Algorithms for Energy Systems Optimal Scheduling

* This code accompanies the paper <i>Performance Comparison of Deep RL Algorithms for Energy Systems Optimal Scheduling</i>, to appear in IEEE PES ISGT EUROPE 2022.
# Abstract 
* Taking advantage of their data-driven and model-free features, Deep Reinforcement Learning (DRL) algorithms have the potential to deal with the increasing level of uncertainty due to the introduction of renewable-based generation. To deal simultaneously with the energy systems' operational cost and technical constraints (e.g, generation-demand power balance) DRL algorithms must consider a trade-off when designing the reward function. This trade-off introduces extra hyperparameters that impact the DRL algorithms' performance and capability of providing feasible solutions. In this paper, a performance comparison of different DRL algorithms, including DDPG, TD3, SAC, and PPO, are presented. We aim to provide a fair comparison of these DRL algorithms for energy systems optimal scheduling problems. Results show DRL algorithms' capability of providing in real-time good-quality solutions, even in unseen operational scenarios, when compared with a mathematical programming model of the energy system optimal scheduling problem. Nevertheless, in the case of large peak consumption, these algorithms failed to provide feasible solutions, which can impede their practical implementation.
# Organization
* Folder "Data" -- Historical and processed data.
* script "agent" and "net"-- General network and agent formulation.
* script "DDPG","SAC","TD3" and "PPO"-- The integration of main process for training, test and plot.
* script "tools"-- General function needed for main process 
* script "random_generator_battery" -- The energy system environment
* Run scripts like DDPG.py after installing all packages. Please have a look for the code structure.
# Dependencies
This code requires installation of the following libraries: ```PYOMO```,```pandas 1.1.4```, ```numpy 1.20.1```, ```matplotlib 3.3.4```, ```pytorch 1.11.0```,  ```math```, you can find more information [at this page](https://ieeexplore.ieee.org/document/9960642).
# Recommended citation
A preprint is available, and you can check this paper for more details  [Link of the paper](https://arxiv.org/abs/2208.00728).
* Paper authors: Hou Shengren, Edgar Mauricio Salazar, Pedro P. Vergara, Peter Palensky
* Accepted for publication at IEEE PES ISGT 2022
* If you use (parts of) this code, please cite the preprint or published paper
@INPROCEEDINGS{9960642,
  author={Shengren, Hou and Salazar, Edgar Mauricio and Vergara, Pedro P. and Palensky, Peter},
  booktitle={2022 IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT-Europe)}, 
  title={Performance Comparison of Deep RL Algorithms for Energy Systems Optimal Scheduling}, 
  year={2022},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ISGT-Europe54678.2022.9960642}}
