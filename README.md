# Age-Structured SIRQVD Model with Viral Load Dynamics

## Research Objective

This project implements a sophisticated epidemiological model that simulates the spread of a disease (like COVID-19) through a population of 1 million people divided into different age groups. The model's novel approach combines age-structured population dynamics with viral load progression to provide more accurate predictions of disease spread and mortality patterns. This integration allows for better understanding of how age-specific factors and viral load dynamics influence disease outcomes and intervention effectiveness.

## Model Architecture

### Core Components

1. **Population Structure**
   - Four distinct age groups with unique epidemiological characteristics
   - Realistic population distribution (0-19: 25%, 20-39: 30%, 40-59: 25%, 60+: 20%)
   - Age-specific parameters for susceptibility, recovery, and mortality

2. **Disease States**
   - Seven-state model (SIRQVDL) capturing disease progression
   - Viral load dynamics integrated with epidemiological states
   - Age-specific transition rates between states

3. **Transmission Dynamics**
   - Age-structured contact matrix
   - Viral load-dependent transmission rates
   - Realistic mixing patterns between age groups

4. **Intervention Framework**
   - Quarantine implementation
   - Vaccination strategies
   - Age-specific intervention effectiveness

## Mathematical Framework

### Differential Equations

The model is based on a system of differential equations that describe how people move between different states:

```
dS/dt = -βSI/N - νS
dI/dt = βSI/N - γI - θI - μI
dQ/dt = θI - γQ - μQ
dR/dt = γ(I + Q)
dV/dt = νS
dD/dt = μ(I + Q)
dL/dt = f(t) - δL
```

### Key Parameters

- β: Transmission rate (0.3 base rate)
- γ: Recovery rate (0.1 base rate)
- θ: Quarantine rate (0.01-0.6)
- ν: Vaccination rate (0.001-0.1)
- μ: Mortality rate (age-dependent)
- f(t): Viral load function
- δ: Viral load decay rate (0.1)

## Implementation

### Data Structures

1. **Contact Matrix**
```python
contact_matrix = np.array([
    [3.0, 1.5, 0.5, 0.2],  # Children/Teens
    [1.5, 2.0, 1.0, 0.3],  # Young Adults
    [0.5, 1.0, 1.5, 0.5],  # Middle Age
    [0.2, 0.3, 0.5, 1.0]   # Elderly
])
```

2. **Age Group Parameters**
```python
age_groups = [
    AgeGroup("0-19", 0.8, 1.2, 0.7, 0.25, 0.0002, 1.2),    # Children/Teens
    AgeGroup("20-39", 1.0, 1.0, 0.8, 0.30, 0.001, 1.8),    # Young Adults
    AgeGroup("40-59", 1.2, 0.9, 0.85, 0.25, 0.005, 2.5),   # Middle Age
    AgeGroup("60+", 1.5, 0.7, 0.9, 0.20, 0.02, 3.5)        # Elderly
]
```

### Simulation Scenarios

```python
scenarios = [
    ("baseline", 0.05, 0.01),           # Baseline scenario
    ("high_quarantine", 0.40, 0.01),    # High quarantine rate
    ("high_vaccination", 0.05, 0.10),   # High vaccination rate
    ("high_both", 0.40, 0.10),         # Both high
    ("low_both", 0.01, 0.001),         # Both low
]
```

## Results and Analysis

### Key Metrics

1. **Epidemiological Indicators**
   - Basic Reproduction Number (R₀)
   - Infection Fatality Ratio (IFR)
   - Years of Life Lost (YLL)
   - Peak Healthcare Demand

2. **Age-Specific Outcomes**
   - Mortality patterns by age group
   - Intervention effectiveness across ages
   - Viral load progression differences
   - Healthcare resource utilization

### Visualization Framework

1. **Time Series Analysis**
   - Disease progression curves
   - Intervention impact timelines
   - Age-specific patterns
   - Viral load trajectories

2. **Comparative Analysis**
   - Scenario comparisons
   - Age group differences
   - Intervention effectiveness
   - Healthcare system impact

## Usage Guide

### Installation
```bash
pip install -r requirements.txt
```

### Running Simulations
```bash
python script.py
```

### Output Analysis
- Results stored in timestamped directories
- Comprehensive JSON analysis files
- Interactive visualization tools
- Comparative scenario analysis

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Seaborn

## Model Validation

### Strengths
- Age-structured population dynamics
- Viral load integration
- Realistic contact patterns
- Comprehensive intervention modeling

### Limitations
- Homogeneous mixing assumption
- Fixed contact patterns
- No healthcare capacity constraints
- Simplified geographic distribution

## Future Directions

1. **Model Enhancements**
   - Healthcare system capacity
   - Geographic distribution
   - Multiple wave modeling
   - Variant tracking

2. **Analysis Improvements**
   - Statistical validation
   - Economic impact modeling
   - Behavioral dynamics
   - Uncertainty analysis

## References

1. Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society of London. Series A, 115(772), 700-721.

2. Anderson, R. M., & May, R. M. (1991). Infectious diseases of humans: dynamics and control. Oxford university press.

3. Ferguson, N. M., et al. (2020). Impact of non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality and healthcare demand. Imperial College London. 