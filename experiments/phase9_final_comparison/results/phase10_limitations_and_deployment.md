# Phase 10 — Limitations and Deployment Considerations

## Project Limitations

### 1. Simulation-based environment
The controller was evaluated in a simulated building-temperature environment.
The thermal dynamics, electricity pricing, occupancy behaviour and outdoor
temperature profiles are therefore representations of building operation rather
than measurements from a live building-management system.

### 2. Limited validation horizon
The final evaluation uses 10 validation profiles, with each profile representing
a 24-hour episode. This provides a controlled unseen evaluation but does not
establish performance across long-term seasonal or yearly operating conditions.

### 3. Simplified thermal dynamics
The building dynamics are represented by the implemented thermal model.
Real buildings may exhibit additional effects such as thermal zones,
building-material differences, solar gains, ventilation, weather uncertainty,
occupant behaviour and equipment-specific dynamics.

### 4. Limited baseline comparison
The principal baseline is a rule-based thermostat controller. Although this
provides a meaningful and interpretable benchmark, it does not establish
superiority over more advanced model-based, optimization-based or alternative
reinforcement-learning controllers.

### 5. Limited training budget
The final multi-seed experiments use approximately 20,000 PPO timesteps per
seed. This is sufficient for the project-scale experiment and produced strong
validation results, but it is not evidence that the policy has reached a
globally optimal solution.

### 6. Synthetic operating profiles
The validation profiles are generated from the project's profile-generation
mechanism. Consequently, generalisation to real occupancy, weather and
electricity-price distributions remains to be demonstrated.

### 7. Single-agent control scope
The project considers HVAC control for the modelled building environment.
It does not explicitly address coordinated control of multiple buildings,
multiple HVAC devices or interaction with a larger energy-management system.

## Real-World Deployment Challenges

### 1. Model mismatch
A policy trained in simulation may encounter building dynamics that differ from
the simulated thermal model. Continuous monitoring and model validation would
therefore be required before deployment.

### 2. Sensor uncertainty
Real deployment depends on temperature and potentially occupancy and
environmental sensors. Measurement noise, missing readings, calibration errors
and sensor failures could affect policy decisions.

### 3. Actuator constraints
Real HVAC equipment has physical operating limits, minimum on/off durations,
ramping constraints, response delays and equipment-specific control interfaces.
These constraints would need to be represented explicitly before deployment.

### 4. Safety and comfort guarantees
Although the final validation achieved zero recorded safety violations and
approximately 99.58% comfort-band coverage, simulated guarantees do not
automatically translate into real-world guarantees. Independent safety
constraints and supervisory control would therefore be appropriate.

### 5. Changing operating conditions
Building occupancy, weather, electricity prices and equipment performance can
change over time. A deployed policy may therefore require monitoring,
revalidation and potentially periodic retraining.

### 6. Integration with building-management systems
A practical deployment would need reliable integration with existing building
automation or energy-management infrastructure, including communication,
control permissions, logging and fault handling.

### 7. Human oversight and fallback control
A real HVAC system should retain a safe fallback controller or supervisory
mechanism so that abnormal policy actions, sensor failures or unexpected
building conditions do not compromise occupant comfort or equipment safety.

### 8. Computational and operational reliability
The deployed controller would need predictable inference latency, reliable
software execution and robust handling of missing or invalid observations.

### 9. Cost and energy objective calibration
The relative weighting of energy, electricity cost, comfort and safety in the
reward function would need to reflect the priorities and operating economics
of the target building rather than being assumed universally applicable.
