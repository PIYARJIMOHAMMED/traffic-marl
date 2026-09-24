\# API and Data Contract



\## Control Request



```json

{

&#x20; "intersection": "A",

&#x20; "proposed\_action": 1,

&#x20; "green\_timer": 20,

&#x20; "in\_yellow": false

}

{

&#x20; "intersection": "A",

&#x20; "proposed\_action": 1,

&#x20; "approved\_action": 1,

&#x20; "valid\_action": true,

&#x20; "minimum\_green\_ok": true,

&#x20; "yellow\_transition\_ok": true,

&#x20; "overridden": false,

&#x20; "safety\_status": "approved"

}

{

&#x20; "intersection": "A",

&#x20; "proposed\_action": 1,

&#x20; "safety\_approved\_action": 1,

&#x20; "final\_action": 1,

&#x20; "human\_approval\_required": false,

&#x20; "human\_approved": true,

&#x20; "status": "approved"

}

{

&#x20; "policy": "ppo\_marl",

&#x20; "scenario": "baseline",

&#x20; "episode": 1,

&#x20; "reward": 302.148,

&#x20; "waiting": 4.588,

&#x20; "queue": 37.447,

&#x20; "speed": 7.097,

&#x20; "throughput": 356.1

}

