# Age-Structured SIRQVD Model with Viral Load Dynamics

## Overview

This project implements a sophisticated epidemiological model that simulates the spread of a disease (like COVID-19) through a population divided into different age groups. The model tracks how the disease spreads, how people recover, and unfortunately, how some people die from the disease. It's particularly useful for understanding how different age groups are affected and how public health measures like quarantine and vaccination can help control the outbreak.

### In Plain English

Think of this model as a virtual city where:
- People are divided into different age groups (kids, young adults, middle-aged, and elderly)
- The disease spreads when people come into contact with each other
- Some people get sicker than others, especially older people
- We can try to control the outbreak by:
  - Isolating sick people (quarantine)
  - Giving people vaccines
  - Understanding how the amount of virus in someone affects how sick they get

## Model Components

### 1. Population Structure

The model divides the population into four age groups:
- 0-19 years (Children/Teens)
- 20-39 years (Young Adults)
- 40-59 years (Middle Age)
- 60+ years (Elderly)

Each age group has different characteristics:
- How likely they are to get infected
- How well they recover
- How well vaccines work for them
- How likely they are to die if infected
- How much the amount of virus affects their health

### 2. Disease States

The model tracks seven different states for each person:
- **S**: Susceptible (can get infected)
- **I**: Infected (currently sick)
- **Q**: Quarantined (isolated to prevent spread)
- **R**: Recovered (got better)
- **V**: Vaccinated (protected)
- **D**: Dead (died from the disease)
- **L**: Viral Load (how much virus is in their body)

### 3. Disease Spread

The disease spreads through:
- Contact between people in different age groups
- The amount of virus in infected people
- How susceptible different age groups are
- How much people interact with each other

### 4. Public Health Interventions

The model includes two main ways to control the outbreak:
- **Quarantine**: Isolating sick people to prevent spread
- **Vaccination**: Protecting people before they get sick

## Technical Details

### Mathematical Formulation

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

Where:
- β: Transmission rate (0.3 base rate)
- γ: Recovery rate (0.1 base rate)
- θ: Quarantine rate (0.01-0.6)
- ν: Vaccination rate (0.001-0.1)
- μ: Mortality rate (age-dependent)
- f(t): Viral load function
- δ: Viral load decay rate (0.1)

### Age-Structured Contact Matrix

The model uses a contact matrix to represent interactions between age groups:

```python
contact_matrix = np.array([
    [3.0, 1.5, 0.5, 0.2],  # Children/Teens
    [1.5, 2.0, 1.0, 0.3],  # Young Adults
    [0.5, 1.0, 1.5, 0.5],  # Middle Age
    [0.2, 0.3, 0.5, 1.0]   # Elderly
])
```

### Viral Load Dynamics

The viral load follows a three-phase pattern with specific mathematical formulation:

1. **Incubation** (0-3 days):
   ```python
   L(t) = L₀ * exp(αt)
   where α = 0.5, L₀ = 0.1
   ```

2. **Peak** (3-8 days):
   ```python
   L(t) = L_max
   where L_max = 1.0
   ```

3. **Decline** (8+ days):
   ```python
   L(t) = L_max * exp(-δ(t - t_peak))
   where δ = 0.1, t_peak = 8
   ```

### Age-Specific Parameters

Each age group has specific epidemiological characteristics:

```python
age_groups = [
    AgeGroup("0-19", 0.8, 1.2, 0.7, 0.25, 0.0002, 1.2),    # Children/Teens
    AgeGroup("20-39", 1.0, 1.0, 0.8, 0.30, 0.001, 1.8),    # Young Adults
    AgeGroup("40-59", 1.2, 0.9, 0.85, 0.25, 0.005, 2.5),   # Middle Age
    AgeGroup("60+", 1.5, 0.7, 0.9, 0.20, 0.02, 3.5)        # Elderly
]
```

### Simulation Parameters

The model includes various intervention scenarios with extreme values:

```python
scenarios = [
    ("baseline", 0.05, 0.002),
    ("high_quarantine", 0.4, 0.002),      # 80% quarantine rate
    ("high_vaccination", 0.05, 0.05),     # 10% vaccination rate
    ("high_both", 0.4, 0.05),            # Both high
    ("low_both", 0.01, 0.001),           # Minimal interventions
    ("extreme_quarantine", 0.6, 0.002),   # 120% quarantine rate
    ("extreme_vaccination", 0.05, 0.1),   # 20% vaccination rate
    ("extreme_both", 0.6, 0.1)           # Both extreme
]
```

### Numerical Integration

The model uses the `odeint` solver from SciPy with:
- Time steps: 1000 points over 180 days
- Relative tolerance: 1e-6
- Absolute tolerance: 1e-6

### Analysis Metrics

The model calculates several key metrics:

1. **Basic Reproduction Number (R₀)**:
   ```python
   R₀ = β/γ * (1 - θ/(γ + θ + μ))
   ```

2. **Infection Fatality Ratio (IFR)**:
   ```python
   IFR = D/(I + Q + R + D)
   ```

3. **Years of Life Lost (YLL)**:
   ```python
   YLL = Σ(D_i * (LE - age_midpoint_i))
   where LE = 80 years
   ```

4. **Peak Healthcare Demand**:
   ```python
   peak_demand = max(I + Q)
   ```

### Output Analysis

The model generates detailed analysis including:
- Time series of all compartments
- Age-specific mortality curves
- Viral load trajectories
- Intervention effectiveness
- Healthcare system burden
- Economic impact estimates

## Analysis and Output

The model generates several types of analysis:

### 1. Individual Scenario Analysis
- Cumulative deaths by age group
- Daily death rates
- Viral load impact on mortality
- Case fatality rates
- Peak daily deaths
- Average viral loads

### 2. Comparison Analysis
- Total deaths across scenarios
- Peak daily deaths comparison
- Infection fatality ratios
- Age-specific mortality rates
- Years of life lost

### Output Files
- **Visualizations**: PNG files showing different aspects of the outbreak
- **JSON Analysis**: Detailed numerical results
- **Parameter Files**: Settings used for each simulation
- **Comparison Summary**: Results across all scenarios

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python script.py
```

3. View results in the `output` directory:
- Each simulation creates a timestamped directory
- Comparison visualizations are in the `comparison` subdirectory
- Detailed analysis is available in JSON format

## Model Limitations

1. **Simplifications**:
   - Assumes homogeneous mixing within age groups
   - Doesn't account for healthcare capacity
   - Ignores geographic distribution
   - Doesn't consider comorbidities

2. **Assumptions**:
   - Fixed contact patterns
   - Constant intervention rates
   - No reinfection
   - No waning immunity

## Future Improvements

1. **Model Enhancements**:
   - Add healthcare system capacity
   - Include geographic distribution
   - Model multiple waves
   - Account for variants

2. **Analysis Improvements**:
   - Add statistical tests
   - Include economic impact
   - Model behavioral changes
   - Add uncertainty analysis

## References

1. Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society of London. Series A, 115(772), 700-721.

2. Anderson, R. M., & May, R. M. (1991). Infectious diseases of humans: dynamics and control. Oxford university press.

3. Ferguson, N. M., et al. (2020). Impact of non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality and healthcare demand. Imperial College London. 