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
The [`ocsf/`](https://github.com/saarcohen30/ocsf/tree/main/ocsf) sub-directory consists of a module for each algorithm, whose execution performs the required testbed. 

### Important Flags
- `-e` -- Specifies the tolerance for GREEDY.
