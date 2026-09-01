# PPO Building Temperature Control

**DSCD 614 --- Reinforcement Learning \| University of Ghana**\
**Project:** Implementation of Proximal Policy Optimization for
Energy-Efficient Building Temperature Control

This repository contains the reproducible implementation, experiments,
validation evidence, and report figures for the PPO-based building
temperature-control study.

The project formulates HVAC control as a sequential decision-making
problem. PPO receives a seven-dimensional continuous observation and
produces a continuous HVAC action in `[-1, 1]`, while the reward
balances energy use, electricity cost, thermal discomfort, and severe
temperature violations.
------------------------------------------------------------------------

## 1. Repository Structure

``` text
PPO-Building-Temperature-Control/
│
├── README.md
├── requirements.txt
├── config/
│   └── environment_config.yaml
│
├── environment/
│   ├── __init__.py
│   ├── profiles.py
│   ├── thermal_model.py
│   ├── reward.py
│   └── building_env.py
│
├── baseline/
│   └── thermostat.py
│
├── data/
│   └── profiles/
│       └── validation_day_001.csv
│
├── notebooks/
│   ├── 01_environment_validation.ipynb
│   ├── 02_baseline_analysis.ipynb
│   ├── 03_ppo_training_analysis.ipynb
│   ├── 04_final_comparative_evaluation.ipynb
│   └── 05_project_compliance_and_final_review.ipynb
│
└── experiments/
    ├── ppo_pilot/
    │   ├── logs/
    │   ├── models/" Contains the Checkpoints"
    │   └── evaluation/
    │
    ├── phase9_final_comparison/
    │   ├── data/
    │   └── results/
    │
    └── phase10_report_evidence/
        ├── figures/
        ├── *.csv
        └── *.txt
```

The `phase10_report_evidence/` directory contains the preserved evidence
used for the final report figures and aggregate results.

------------------------------------------------------------------------

## 2. Environment and Experimental Setup

### Building environment

-   Single-zone simulated building
-   24 hourly steps per episode
-   Initial indoor temperature: `22°C`
-   Comfort setpoint: `22°C`
-   Comfort band: `21–23°C`
-   Severe safety limits: `18–26°C`
-   Continuous HVAC action: `[-1, 1]`
    -   negative = cooling
    -   zero = HVAC off
    -   positive = heating
-   Observation dimension: `7`

### Observation variables

1.  Indoor temperature
2.  Outdoor temperature
3.  Comfort setpoint
4.  Electricity price
5.  Occupancy
6.  `sin(hour)`
7.  `cos(hour)`

Temperature observations are scaled by 40; occupancy is bounded in
`[0,1]`; cyclic time features lie in `[-1,1]`.

### Thermal dynamics

The physics-inspired model uses:

-   Outdoor-temperature coupling: `0.08`
-   HVAC effect: `3.0°C` per unit action
-   Occupancy heat gain: `0.15°C`
-   Disturbance standard deviation: `0.0°C`

### Reward

The reward combines:

-   HVAC energy penalty: `0.30`
-   Electricity-cost penalty: `0.20`
-   Thermal-discomfort penalty: `0.40`
-   Severe-temperature violation penalty: `1.00`

The implementation is in `environment/reward.py`.

------------------------------------------------------------------------

## 3. Installation

### Option A --- Google Colab

The notebooks were designed for a Google Drive/Colab workflow.

1.  Upload or clone the complete project folder to Google Drive.
2.  Mount Google Drive in Colab.
3.  Set the project root used by the notebooks to:

``` python
PROJECT_ROOT = Path(
    "/content/drive/MyDrive/PPO-Building-Temperature-Control"
)
```

4.  Install the dependencies:

``` python
!pip install -r "/content/drive/MyDrive/PPO-Building-Temperature-Control/requirements.txt"
```

5.  Open the notebooks from the `notebooks/` directory.

### Option B --- Local Python environment

From the project root:

``` bash
python -m venv .venv
```

Activate the environment.

**Windows:**

``` bash
.venv\Scripts\activate
```

**Linux/macOS:**

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Launch Jupyter:

``` bash
jupyter notebook
```

or:

``` bash
jupyter lab
```

------------------------------------------------------------------------

## 4. Reproducible Workflow

Run the notebooks in the following order.

  ------------------------------------------------------------------------------------------------
  Order                   Notebook                                         Purpose
  ----------------------- ------------------------------------------------ -----------------------
  1                       `01_environment_validation.ipynb`                Generate and validate
                                                                           building-condition
                                                                           profiles and verify the
                                                                           custom environment

  2                       `02_baseline_analysis.ipynb`                     Implement and evaluate
                                                                           the rule-based
                                                                           thermostat baseline

  3                       `03_ppo_training_analysis.ipynb`                 Configure, train,
                                                                           validate, and analyse
                                                                           PPO

  4                       `04_final_comparative_evaluation.ipynb`          Perform controlled
                                                                           PPO-versus-thermostat
                                                                           comparison

  5                       `05_project_compliance_and_final_review.ipynb`   Audit the completed
                                                                           experiment and
                                                                           regenerate final report
                                                                           evidence and figures
  ------------------------------------------------------------------------------------------------

### Important

Notebook 03 contains several development and verification phases,
including the 5k pilot, extended training, multi-seed stability
analysis, and final PPO evidence extraction.

The final report uses the **preserved final evidence** in:

``` text
experiments/phase10_report_evidence/
```


------------------------------------------------------------------------

# 5. Training PPO

The final multi-seed experiment uses the following independent random
seeds:

``` text
42
123
2024
```

The final training budget is approximately 20,000 environment timesteps
per seed. Because PPO collects complete rollout batches of 256 steps,
the recorded final training length is:

``` text
20,224 timesteps per seed
```

The PPO configuration is:

  Hyperparameter                      Value
  ---------------------------- ------------
  Learning rate                    `0.0003`
  Rollout steps                       `256`
  Batch size                           `64`
  PPO epochs                           `10`
  Discount factor γ                  `0.99`
  GAE λ                              `0.95`
  Clip range                         `0.20`
  Entropy coefficient                `0.01`
  Value-function coefficient         `0.50`
  Maximum gradient norm              `0.50`
  Network                        `[64, 64]`

### Training procedure

For a clean reproduction:

1.  Run Notebook 01.
2.  Run Notebook 02.
3.  Run Notebook 03 from the beginning.
4.  Allow the notebook to complete its environment checks, pilot
    training, extended training, and multi-seed training phases.
5.  Preserve the generated models and TensorBoard logs under:

``` text
experiments/ppo_pilot/
```

The training reward evidence is extracted from the actual TensorBoard
`rollout/ep_rew_mean` observations rather than reconstructed from
evaluation metrics.

------------------------------------------------------------------------

# 6. Evaluation

## 6.1 PPO evaluation

The final analysis evaluates independent PPO runs using deterministic
action selection:

``` python
model.predict(
    observation,
    deterministic=True
)
```

This disables stochastic exploration during evaluation.

The final report aggregates performance across the three independent
seeds rather than selecting the best run.

Primary metrics:

-   Total HVAC energy
-   Electricity cost
-   Mean temperature deviation
-   Total discomfort
-   Comfort-band coverage
-   Safety violations
-   Cumulative reward
-   Peak HVAC energy

## 6.2 Thermostat baseline

The baseline is implemented in:

``` text
baseline/thermostat.py
```

Configuration:

``` text
Setpoint = 22°C
Tolerance = 1°C
```

The controller:

-   heats at `+1` when temperature is below `21°C`;
-   remains off at `0` inside the comfort band;
-   cools at `-1` when temperature is above `23°C`.

For a fair comparison, PPO and the thermostat are evaluated using the
same environment, validation profiles, episode length, metric
definitions, and evaluation procedure.

## 6.3 Held-out evaluation

The final report evidence uses **30 unseen validation profiles**, each
containing **24 hourly observations**.

For PPO:

``` text
3 seeds × 30 profiles × 24 hours
= 2,160 hourly observations
```

The thermostat is evaluated on the same validation-profile set.

> Some earlier Phase 9 files in `experiments/phase9_final_comparison/`
> contain the original 10-profile evaluation used during development and
> audit. They are intentionally retained for traceability. The final
> report figures use the preserved Phase 10 report-evidence files.

------------------------------------------------------------------------

# 7. Reproducing Every Figure in the Report

All five submitted report figures are generated in:

``` text
notebooks/05_project_compliance_and_final_review.ipynb
```

The generated images are saved to:

``` text
experiments/phase10_report_evidence/figures/
```

## Figure 1 --- PPO Training Return Across Random Seeds

**Report figure:** PPO mean training return with ±1 standard deviation.

**Source evidence:**

``` text
experiments/phase10_report_evidence/
└── ppo_training_return_mean_std.csv
```

**Notebook:** `05_project_compliance_and_final_review.ipynb`

**Figure-generation section:** `REPORT FIGURE 1 — PPO TRAINING RETURN`

The figure is generated from the actual TensorBoard-derived columns:

``` text
step
mean_reward
std_reward
```

Output:

``` text
figure_1_ppo_training_return_mean_sd.png
```

### Reproduction

1.  Ensure `ppo_training_return_mean_std.csv` exists.
2.  Open Notebook 05.
3.  Run the Figure 1 section.
4.  Confirm that the figure is written to:

``` text
experiments/phase10_report_evidence/figures/
```

------------------------------------------------------------------------

## Figure 2 --- PPO vs Thermostat: Energy and Electricity Cost

**Report figure:** comparison of PPO and thermostat total energy and
electricity cost.

**Source evidence:**

``` text
experiments/phase10_report_evidence/
└── ppo_vs_thermostat_30_episode_comparison.csv
```

**Notebook:** `05_project_compliance_and_final_review.ipynb`

**Figure-generation section:**
`REPORT FIGURE 2 — PPO VS THERMOSTAT — ENERGY AND ELECTRICITY COST`

The figure uses:

``` text
total_energy
electricity_cost
```

PPO bars use the three-seed mean and ±1 SD; the thermostat value is
evaluated on the same validation profiles.

Output:

``` text
figure_2_ppo_vs_thermostat_energy_cost.png
```

### Reproduction

1.  Ensure the comparison CSV exists.
2.  Run the Figure 2 section in Notebook 05.
3.  Verify the generated PNG in `phase10_report_evidence/figures/`.

------------------------------------------------------------------------

## Figure 3 --- PPO vs Thermostat: Thermal Comfort

**Report figure:** comparison of:

-   comfort-band coverage;
-   mean temperature deviation;
-   total discomfort.

**Source evidence:**

``` text
experiments/phase10_report_evidence/
└── ppo_vs_thermostat_30_episode_comparison.csv
```

**Notebook:** `05_project_compliance_and_final_review.ipynb`

**Figure-generation section:**
`REPORT FIGURE 3 — PPO VS THERMOSTAT — COMFORT PERFORMANCE`

Output:

``` text
figure_3_ppo_vs_thermostat_comfort.png
```

Safety violations are reported separately because both controllers
recorded zero violations.

### Reproduction

Run the Figure 3 section after the Figure 2 section has loaded the
comparison evidence.

------------------------------------------------------------------------

## Figure 4 --- Representative 24-Hour PPO Temperature Trajectory

**Report figure:** representative indoor-temperature trajectory for one
complete 24-hour validation episode.

The report uses:

``` text
Seed = 123
Profile = validation_day_0001
```

**Source evidence:**

``` text
experiments/phase10_report_evidence/
└── ppo_30_episode_validation.csv
```

The selected episode is filtered by:

``` python
seed == 123
profile_id == "validation_day_0001"
```

The notebook verifies:

-   exactly 24 observations;
-   hours `0–23`;
-   non-missing indoor temperatures;
-   valid HVAC actions in `[-1, 1]`.

Output:

``` text
figure_4_representative_ppo_temperature_trajectory.png
```

### Reproduction

1.  Ensure `ppo_30_episode_validation.csv` exists.
2.  Run the Figure 4 section in Notebook 05.
3.  The notebook selects Seed 123 / `validation_day_0001`.
4.  The resulting figure is saved automatically.

This figure is **illustrative** and is not used to calculate the
aggregate performance statistics.

------------------------------------------------------------------------

## Figure 5 --- Representative PPO HVAC Control Actions

**Report figure:** the continuous HVAC action sequence corresponding to
the representative Seed 123 episode.

It uses the same trajectory created for Figure 4.

Output:

``` text
figure_5_representative_ppo_hvac_action.png
```

### Reproduction

Run the Figure 5 section immediately after Figure 4 in Notebook 05.

The action sequence is plotted directly from the preserved `action`
column and therefore does not require retraining the model.

------------------------------------------------------------------------

# 8. Fast Figure-Only Reproduction

If the goal is only to regenerate the five figures for the report, **do
not retrain PPO**.

Use the preserved evidence already included in:

``` text
experiments/phase10_report_evidence/
```

Open:

``` text
notebooks/05_project_compliance_and_final_review.ipynb
```

and run the report-figure sections in this order:

``` text
Figure 1
    ↓
Figure 2
    ↓
Figure 3
    ↓
Figure 4
    ↓
Figure 5
```

This reproduces the report graphics from the preserved experimental
evidence.

------------------------------------------------------------------------

# 9. Key Evidence Files

  -----------------------------------------------------------------------------------------
  File                                                  Purpose
  ----------------------------------------------------- -----------------------------------
  `ppo_training_return_mean_std.csv`                    TensorBoard-derived multi-seed
                                                        training-return evidence

  `ppo_vs_thermostat_30_episode_comparison.csv`         Final PPO-versus-thermostat
                                                        comparison data

  `ppo_30_episode_validation.csv`                       PPO hourly validation trajectories

  `thermostat_30_episode_validation.csv`                Thermostat hourly validation
                                                        trajectories

  `ppo_30_episode_metrics.csv`                          PPO episode-level evaluation
                                                        metrics

  `thermostat_30_episode_metrics.csv`                   Thermostat episode-level evaluation
                                                        metrics

  `ppo_30_episode_seed_summary.csv`                     PPO seed-level summary

  `phase10_multiseed_mean_std.csv`                      Mean/std multi-seed performance
                                                        evidence

  `ppo_seed_run_mapping_audit.txt`                      Audit trail linking PPO runs and
                                                        seeds

  `phase10_final_requirements_traceability_final.csv`   Final project
                                                        requirement-to-evidence
                                                        traceability
  -----------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 10. Verification Checklist

Before considering a reproduction successful, verify:

### Environment

-   [ ] Environment imports successfully.
-   [ ] Observation shape is `(7,)`.
-   [ ] Action space is `[-1, 1]`.
-   [ ] Each profile contains exactly 24 hourly observations.
-   [ ] Comfort band is `21–23°C`.
-   [ ] Safety limits are `18–26°C`.

### Training

-   [ ] PPO trains with seeds `42`, `123`, and `2024`.
-   [ ] Each final run reaches `20,224` timesteps.
-   [ ] TensorBoard logs contain actual `rollout/ep_rew_mean`
    observations.
-   [ ] Training reward evidence is extracted from TensorBoard.

### Evaluation

-   [ ] PPO evaluation uses deterministic actions.
-   [ ] Validation profiles are held out from training.
-   [ ] PPO and thermostat use identical evaluation conditions.
-   [ ] All required metrics are present.
-   [ ] Mean and standard deviation are reported across PPO seeds.

### Figures

-   [ ] Figure 1 is generated from TensorBoard-derived reward evidence.
-   [ ] Figure 2 uses the final PPO/thermostat comparison evidence.
-   [ ] Figure 3 uses the same comparison evidence.
-   [ ] Figure 4 uses Seed 123 / `validation_day_0001`.
-   [ ] Figure 5 uses the same representative trajectory.
-   [ ] All figures are saved at 300 DPI.

------------------------------------------------------------------------

# 11. Reproducibility Notes

### Random seeds

The final PPO experiment uses:

``` text
42
123
2024
```

The profile-generation configuration is frozen through
`environment_config.yaml`.

### Validation

Evaluation is separated from training. The final analysis uses held-out
validation profiles and deterministic PPO inference.

### Existing artifacts

The repository preserves trained models, TensorBoard logs, evaluation
CSV files, audit files, and final figures. These artifacts allow the
final report evidence to be inspected without rerunning the complete
training experiment.

### Development versus final evidence

The project contains earlier experimental artifacts because the workflow
was developed incrementally. These files are retained rather than
deleted so that the experimental history remains auditable.

For reproducing the **submitted report**, use:

``` text
experiments/phase10_report_evidence/
```

and:

``` text
notebooks/05_project_compliance_and_final_review.ipynb
```

------------------------------------------------------------------------

# 12. Project Outcome

Under the evaluated simulation conditions, the final PPO controller
demonstrated a substantially better energy--comfort trade-off than the
implemented thermostat baseline.

The final report reports:

-   **52.38% lower energy consumption**
-   **50.53% lower electricity cost**
-   **98.64% lower total discomfort**
-   **98.33% comfort-band coverage**
-   **0 safety violations**

These values should be treated as experimental findings for the defined
simulated environment and validation profiles, not as evidence of
immediate real-world HVAC deployment readiness.

------------------------------------------------------------------------

## 13. Academic and Experimental Integrity

The project distinguishes between:

1.  **Implementation evidence** --- source code and notebooks;
2.  **Training evidence** --- PPO models and TensorBoard logs;
3.  **Evaluation evidence** --- held-out validation trajectories and
    metrics;
4.  **Report evidence** --- preserved CSV summaries and generated
    figures.



The project therefore supports a reproducible chain:

``` text
Environment
    ↓
PPO Training
    ↓
TensorBoard / Model Artifacts
    ↓
Held-Out Evaluation
    ↓
Metric Aggregation
    ↓
Report Evidence
    ↓
Figures
    ↓
Final Report
```

------------------------------------------------------------------------

## 14. References

-   Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.
    (2017). *Proximal Policy Optimization Algorithms*.
-   Sierla, S., Ihasalo, H., & Vyatkin, V. (2022). *A review of
    reinforcement learning applications to control of heating,
    ventilation and air conditioning systems*. Energies, 15(10), 3526.
-   Al Sayed, K., Boodi, A., Sadeghian Broujeny, R., & Beddiar, K.
    (2024). *Reinforcement learning for HVAC control in intelligent
    buildings: A technical and conceptual review*. Journal of Building
    Engineering, 95, 110085.
-   Togashi, E. (2025). *Reward function design in reinforcement
    learning for HVAC control: A review of thermal comfort and energy
    efficiency trade-offs*. Energy and Buildings, 348, 116439.
