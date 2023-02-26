# Online Coalitional Skill Formation
Code for implementation of the following assignment algorithms for our novel framework, termed as <i>online coalitional skill formation</i> (<b>OCSF</b>), for handling online task allocation from a standpoint of coalition formation:
- `GREEDY` - A greedy algorithm that assigns an agent to a task as long as the task's requirements and budget are not violated, and regardless of the (known or unknown) agent distribution.
- `OCSF-CMDP` - We show that the constraints incurred by the tasks' requirements allows us to formulate the system as constrained MDPs (CMDPs). When the agents' distribution is <i>known</i>, we prove that our goal is maximizing the rate at which agents are assigned to each task while respecting their requirements. Based on the CMDP's optimal and stationary policy, OCSF-CMDP assigns agents to tasks until their budgets are reached.
- `OptCMDP` and `TuneOptOCSF` - If the distribution is <i>unknown</i>, those algorithms that learn it online.

If any part of this code is used, the following paper must be cited: 

Saar Cohen and Noa Agmon. Online Coalitional Skill Formation. <em>In AAMAS'23: Proceedings of the 22th International Conference on Autonomous Agents and Multiagent Systems, 2023</em> (to appear).

## Dependencies
Evaluations were conducted using a 12GB NVIDIA Tesla K80 GPU, and implemented in Python3 with:
- PyTorch v2.6.0 (The implementation appears [here](https://github.com/saarcohen30/GrPR2-A/tree/main/grpr2-ch-colab)).
- PyTorch v1.12.0, which is suitable for environment without support of higher versions of PyTorch (The implementation appears [here](https://github.com/saarcohen30/GrPR2-A/tree/main/grpr2-ch)).

**Note:** Each framework contains a `requirements.txt` file, which specifies the modules required for its execution on the respective PyTorch version. For instance, for the imlementation suitable for PyTorch v2.6.0, the `requirements.txt` file cotains a script which aims at downloading all the modules required for its execution on Google's Colaboratory.

## The Cooperative Navigation Task
In this task of the Particle World environment, `n` agents must cooperate through physical actions to reach a set of $n$ landmarks. Agents observe the relative positions of nearest agents and landmarks, and are collectively rewarded based on the proximity of any agent to each landmark. In other words, the agents have to "cover" all of the landmarks. Further, the agents occupy significant physical space and are penalized when colliding with each other. Our agents learn to infer the landmark they must cover, and move there while avoiding other agents. Though the environment holds a continuous state space, agents' actions space is discrete, and given by all possible directions of movement for each agent `{up, down, left, right, stay}`. Given an interaction graph, we augment this task for enabling local information sharing between neighbors, as outlined subsequently.

## Execution
The [`grpr2-ch/`](https://github.com/saarcohen30/GrPR2-CH/tree/main/grpr2-ch) and [`grpr2-ch-colab/`](https://github.com/saarcohen30/GrPR2-CH/tree/main/grpr2-ch-colab) sub-directories consist of the `main.py` module, whose execution performs the required testbed. Specifically, the following executions are possible:
- `python grpr2-ch/main.py simple_spread_local maac` or `python grpr2-ch-colab/main.py simple_spread_local maac --train_graph True` - For a setup of `n=4` agents.
- `python regma/regma.py simple_spread_hetero maac` or `python grpr2-ch-colab/main.py simple_spread_hetero maac` - For a setup of `n=8` agents.

### Important Flags
- `--train_graph` -- In both setups (of either `n=4` or `n=8` agents), one can possibly decide whether to train the graph reasoning policy or not. After specifying `--train_graph true` upon the execution, the graph reasoning policy will be trained. By default, the graph reasoning policy will **not** be trained.
- `--pretrained_graph` -- In both setups (of either `n=4` or `n=8` agents), one can possibly decide whether to utilize a pre-trained graph reasoning policy, which shall be stored in a file named as `local_graph.pt`. For this sake, the `--pretrained_graph` flag shall be set to true by specifying `--pretrained_graph true` upon the execution. By default, a pretrained graph reasoning policy will **not** be incorporated.
- `--model_names_setting` - This flag specifies the names of the model to be trained. The possible models for initializing an self-play enviroment are as follows:

| The Flag's Argument | The Model's Description |
| ------------- | ------------- |
| GrPR2AC`k`_GrPR2AC`k`  | For level-`k` GrPR2-CH agents |
| PR2AC`k`_PR2AC`k`  | For level-`k` GrPR2-L agents |
| DDPG_DDPG | For DDPG independent learners, which are regarded as having level-0 reasoning. |
| MADDPG_MADDPG | For MADDPG agents, which are regarded as having level-0 reasoning. |
| DDPG-ToM_DDPG-ToM | For DDPG agents with a level-1 [Theory-of-Mind model](http://proceedings.mlr.press/v80/rabinowitz18a/rabinowitz18a.pdf) that captures the dependency of an agent’s policy on opponents’ mental states (DDPG-ToM). |
| DDPG-OM_DDPG-OM | For DDPG agents with a level-0 model of [opponent modeling](http://proceedings.mlr.press/v48/he16.pdf), which is implemented by augmenting DDPG with an opponent module (DDPG-OM) that predicts opponents' behaviors in future states. |
