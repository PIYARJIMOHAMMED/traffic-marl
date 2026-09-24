\# Research Conclusion



\## Research Question



Does network-level multi-agent coordination improve a learned traffic signal

controller compared with a local-only learned controller?



\## Finding



Yes, within the tested simulation environment, PPO-MARL improved upon PPO-Local

in most reported metrics.



PPO-MARL achieved:



\- 4.93% higher reward

\- 1.73% lower waiting time

\- 2.33% lower queue length

\- 0.86% higher average speed



However, PPO-MARL throughput was 2.55% lower than PPO-Local.



\## Interpretation



The results provide evidence that graph-based network information and

multi-agent coordination can improve several aspects of learned traffic

control, although the current policy has a throughput trade-off.



The strongest classical baselines remained superior in aggregate traffic

performance.



\## Contribution



The project combines:



\- SUMO-based traffic simulation

\- multi-agent PPO

\- graph-based observations

\- neighbour communication

\- cooperative reward design

\- scenario-based evaluation

\- safety validation

\- human-governance control

\- audit logging



\## Limitation



The learned controller has not demonstrated consistent superiority over strong

rule-based controllers.



Therefore the contribution should be presented as a validated MARL prototype

and coordination study rather than a claim of production superiority.



\## Future Work



Future work should investigate:



1\. longer MARL training

2\. improved reward balancing

3\. throughput-aware reward shaping

4\. larger traffic networks

5\. decentralized execution

6\. domain randomization

7\. real-world traffic-data calibration

8\. hardware-in-the-loop testing

9\. field validation

10\. improved uncertainty estimation

