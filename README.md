# Online Coalitional Skill Formation
Code for implementation of the following assignment algorithms for our novel framework, termed as <i>online coalitional skill formation</i> (<b>OCSF</b>), for handling online task allocation from a standpoint of coalition formation:
- `GREEDY` - A greedy algorithm that assigns an agent to a task as long as the task's requirements and budget are not violated, and regardless of the (known or unknown) agent distribution.
- `OCSF-CMDP` - We show that the constraints incurred by the tasks' requirements allows us to formulate the system as constrained MDPs (CMDPs). When the agents' distribution is <i>known</i>, we prove that our goal is maximizing the rate at which agents are assigned to each task while respecting their requirements. Based on the CMDP's optimal and stationary policy, OCSF-CMDP assigns agents to tasks until their budgets are reached.
- `OptCMDP` and `TuneOptOCSF` - If the distribution is <i>unknown</i>, those algorithms that learn it online.

If any part of this code is used, the following paper must be cited: 

Saar Cohen and Noa Agmon. Online Coalitional Skill Formation. <em>In AAMAS'23: Proceedings of the 22th International Conference on Autonomous Agents and Multiagent Systems, 2023</em> (to appear).

## Dependencies
Evaluations were implemented in Python3 with:
- pulp-2.6.0

## Online Coalitional Skill Formation
Online coalitional skill formation (OCSF) handles online task allocation from a standpoint of coalition formation. In our formalization, there is a set of `m` <i>skills</i> and each agent has a <i>skill vector</i> that expresses her level at mastering each skill. Additionally, an <i>organizer</i> has a fixed set of `k` <i>tasks</i>, each with certain requirements reflecting the desired skill levels essential to complete the task, and the number of agents assigned to each task is limited by some <i>budget</i>. Agents arrive online, and must <i>immediately</i> and <i>irrevocably</i> be assigned to a coalition attending a task upon arrival, if at all. We propose a <i>new</i> skill model for online task allocation, where the set of possible mastering levels for each skill is <i>discrete</i>, and a coalition is evaluated by the extent each skill level is <i>covered</i> by the coalition.

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
