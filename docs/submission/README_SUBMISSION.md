\# MAI-07 — Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic Control



\## 1. Project Summary



This project develops a simulation-based multi-agent reinforcement learning

system for adaptive traffic signal control across a connected four-intersection

traffic network.



The system uses SUMO for traffic simulation and PPO for learning-based traffic

signal control.



The proposed MARL controller uses graph-based network observations so that each

intersection can use local traffic information together with information from

neighbouring intersections.



\## 2. System



Network:

\- Four intersections: A, B, C and D

\- 2x2 connected topology

\- Neighbour relationships:

&#x20; - A -> B, C

&#x20; - B -> A, D

&#x20; - C -> A, D

&#x20; - D -> B, C



Observation:

\- Local traffic features: 15

\- Neighbour features: 15

\- Graph/network representation: 30

\- Four-agent global observation: 120



Action:

\- MultiDiscrete(\[2,2,2,2])

\- One binary control decision per intersection.



\## 3. Learning



Algorithm:

\- Proximal Policy Optimization (PPO)



The final MARL model uses a 120-dimensional observation.



Final model:

rl/checkpoints/ppo\_marl\_120d\_final.zip



Local PPO ablation:

rl/checkpoints/ppo\_marl\_250000\_steps.zip



\## 4. Safety and Governance



The control pipeline includes:



1\. PPO action proposal

2\. Safety validation

3\. Human approval/governance gate

4\. Final action execution

5\. Audit logging



Safety constraints include minimum-green protection and yellow-transition

protection.



\## 5. Evaluation



The final unified experiment contains:



\- 5 policies

\- 10 scenarios

\- 3 episodes per policy/scenario

\- 300 maximum simulation steps

\- 150 total episodes



Policies:

\- Fixed-Time

\- Random

\- Queue-Based

\- PPO-Local

\- PPO-MARL



The final experiment completed:



\- 150/150 episodes

\- 0 runtime errors

\- 0 missing metrics

\- 0 invalid episodes

\- 0 negative-throughput records



\## 6. Final Results



| Policy | Reward | Waiting | Queue | Speed | Throughput |

|---|---:|---:|---:|---:|---:|

| Fixed-Time | 351.895 | 1.738 | 22.427 | 8.163 | 389.500 |

| Queue-Based | 379.018 | 2.873 | 23.407 | 8.155 | 372.300 |

| PPO-MARL | 302.148 | 4.588 | 37.447 | 7.097 | 356.100 |

| PPO-Local | 287.943 | 4.669 | 38.340 | 7.036 | 365.400 |

| Random | 261.332 | 8.706 | 52.465 | 6.354 | 328.933 |



\## 7. Main Research Finding



PPO-MARL improves over PPO-Local in reward, waiting time, queue length and

average speed.



The improvement demonstrates that network-level coordination and graph-based

information can improve a learned traffic-control policy.



However, the classical Fixed-Time and Queue-Based controllers achieved stronger

aggregate traffic-performance results in this experiment.



Therefore, the system is considered a research prototype rather than a

production-ready traffic-control system.



\## 8. Safety Result



The safety/robustness evaluation recorded zero safety violations while the

safety layer applied overrides to proposed unsafe actions.



This demonstrates the operation of the safety-governance layer under the

tested scenarios.



\## 9. Deployment Decision



GO:

\- Research prototype

\- Simulation demonstration

\- Stakeholder evaluation

\- Further experimental development



NO-GO:

\- Direct deployment on real traffic infrastructure without additional

&#x20; validation, training and field testing.



\## 10. Reproducibility



Main evaluation:



python rl\\evaluation\\final\_unified\_evaluation.py



Final results are stored under:



rl/results/final\_comparison/



Important files:



\- final\_unified\_raw.csv

\- final\_unified\_summary.csv

\- final\_unified\_scenario\_comparison.csv

\- final\_unified\_statistics.csv

\- final\_unified\_ranking.csv

\- final\_unified\_results.json

\- final\_unified\_validation.json

