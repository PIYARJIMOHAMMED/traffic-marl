\# System Architecture



\## High-Level Architecture



&#x20;                   +----------------------+

&#x20;                   |   Scenario Manager   |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   |    SUMO Simulator    |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                +-------------+-------------+

&#x20;                |      Traffic State        |

&#x20;                +-------------+-------------+

&#x20;                              |

&#x20;            +-----------------+-----------------+

&#x20;            |                 |                 |

&#x20;            v                 v                 v

&#x20;       Agent A            Agent B           Agent C ... Agent D

&#x20;            |                 |                 |

&#x20;            +-----------------+-----------------+

&#x20;                              |

&#x20;                              v

&#x20;                   Graph-Based Observation

&#x20;                        120 dimensions

&#x20;                              |

&#x20;                              v

&#x20;                        PPO-MARL Policy

&#x20;                              |

&#x20;                              v

&#x20;                      Proposed Actions

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   |   Safety Validator   |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   | Human Approval Gate  |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   | Traffic Controllers  |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   |    Audit Logger      |

&#x20;                   +----------------------+



\## Main Components



\### SUMO

Provides the traffic simulation environment.



\### Agents

Each intersection is represented as an independent traffic-control agent.



\### Observation Builder

Constructs local, neighbouring and graph/network features.



\### PPO-MARL

Produces coordinated traffic signal actions.



\### Safety Validator

Checks proposed actions against signal safety constraints.



\### Human Approval Gate

Supports automatic, approval-required and emergency governance modes.



\### Audit Logger

Stores decisions and governance outcomes for traceability.



\## Data Flow



Traffic state

\-> observation construction

\-> graph/network representation

\-> PPO action

\-> safety validation

\-> governance

\-> signal execution

\-> updated traffic state.

