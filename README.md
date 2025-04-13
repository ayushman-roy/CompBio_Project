# Age-Structured SIRQVD Model with Viral Load Dynamics

## Overview

This project implements a sophisticated epidemiological model that simulates the spread of a disease (like COVID-19) through a population of 1 million people divided into different age groups. The model tracks how the disease spreads, how people recover, and unfortunately, how some people die from the disease. All outputs are presented as percentages of the total population, making it easier to understand the relative impact across different scenarios.

### In Plain English

Think of this model as a virtual city of 1 million people where:
- People are divided into different age groups (kids, young adults, middle-aged, and elderly)
- The disease spreads when people come into contact with each other
- Some people get sicker than others, especially older people
- We can try to control the outbreak by:
  - Isolating sick people (quarantine)
  - Giving people vaccines
  - Understanding how the amount of virus in someone affects how sick they get
- All results are shown as percentages (e.g., "2% of the population died" instead of "20,000 people died")

## Model Components

### 1. Population Structure

The model divides the population of 1 million into four age groups:
- 0-19 years (Children/Teens) - 250,000 people (25%)
- 20-39 years (Young Adults) - 300,000 people (30%)
- 40-59 years (Middle Age) - 250,000 people (25%)
- 60+ years (Elderly) - 200,000 people (20%)

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

All states are tracked as percentages of the total population.

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
    AgeGroup("0-19", 0.8, 1.2, 0.7, 0.25, 0.0002, 1.2),    # Children/Teens (250,000 people)
    AgeGroup("20-39", 1.0, 1.0, 0.8, 0.30, 0.001, 1.8),    # Young Adults (300,000 people)
    AgeGroup("40-59", 1.2, 0.9, 0.85, 0.25, 0.005, 2.5),   # Middle Age (250,000 people)
    AgeGroup("60+", 1.5, 0.7, 0.9, 0.20, 0.02, 3.5)        # Elderly (200,000 people)
]
```

### Simulation Parameters

The model includes various intervention scenarios:

```python
scenarios = [
    ("baseline", 0.05, 0.01),           # Baseline scenario
    ("high_quarantine", 0.40, 0.01),    # High quarantine rate
    ("high_vaccination", 0.05, 0.10),   # High vaccination rate
    ("high_both", 0.40, 0.10),         # Both high
    ("low_both", 0.01, 0.001),         # Both low
]
```

### Numerical Integration

The model uses the `odeint` solver from SciPy with:
- Time steps: 1000 points over 180 days
- Relative tolerance: 1e-6
- Absolute tolerance: 1e-6

### Analysis Metrics

The model calculates several key metrics, all as percentages of the total population:

1. **Basic Reproduction Number (R₀)**:
   ```python
   R₀ = β/γ * (1 - θ/(γ + θ + μ))
   ```

2. **Infection Fatality Ratio (IFR)**:
   ```python
   IFR = D/(I + Q + R + D) * 100  # As percentage
   ```

3. **Years of Life Lost (YLL)**:
   ```python
   YLL = Σ(D_i * (LE - age_midpoint_i))
   where LE = 80 years
   ```

4. **Peak Healthcare Demand**:
   ```python
   peak_demand = max(I + Q) * 100  # As percentage
   ```

### Output Analysis

The model generates detailed analysis including:
- Time series of all compartments (as percentages)
- Age-specific mortality curves (as percentages)
- Viral load trajectories
- Intervention effectiveness (as percentages)
- Healthcare system burden (as percentages)
- Economic impact estimates

## Understanding Percentage-Based Calculations

### Example Calculations

1. **Population Distribution**:
   - Total population: 1,000,000
   - Example: 25% of population in 0-19 age group
   - Calculation: 1,000,000 × 0.25 = 250,000 people

2. **Daily Mortality Rates**:
   - Elderly mortality rate: 2% per day
   - Example: If 100 elderly are infected
   - Calculation: 100 × 0.02 = 2 deaths per day

3. **Vaccination Coverage**:
   - High vaccination rate: 10% per day
   - Example: If 500,000 susceptible
   - Calculation: 500,000 × 0.10 = 50,000 vaccinated per day

4. **Infection Fatality Ratio (IFR)**:
   - Example: 1,000 total infections, 20 deaths
   - Calculation: (20/1000) × 100 = 2% IFR

5. **Years of Life Lost (YLL)**:
   - Example: 100 deaths in 40-59 age group (midpoint 50)
   - Calculation: 100 × (80 - 50) = 3,000 YLL

### Interpreting Percentage Outputs

1. **Population Impact**:
   - 1% of population = 10,000 people
   - 0.1% daily change = 1,000 people affected
   - Small percentages can represent large numbers

2. **Rate Interpretation**:
   - Daily rates are per capita
   - Example: 0.5% daily mortality = 5 deaths per 1000 infected
   - Rates compound over time

3. **Relative Changes**:
   - Doubling from 1% to 2% = 10,000 more people
   - 50% reduction in transmission = half as many new cases
   - Percentage changes show relative impact

4. **Age-Specific Metrics**:
   - Rates are relative to age group size
   - Example: 1% mortality in elderly = 2,000 people
   - Same percentage, different absolute numbers

### Scaling Process Details

1. **Population Scaling**:
   - All initial conditions sum to 100%
   - Each age group starts as fraction of total
   - Example: 25% susceptible = 250,000 people

2. **Rate Conversion**:
   - Raw rates converted to daily percentages
   - Example: 0.3 transmission rate = 30% chance per contact
   - All rates normalized to daily basis

3. **Contact Matrix Scaling**:
   - Contact rates relative to baseline
   - Example: 3.0 means 3× more contacts than baseline
   - Matrix values are multipliers

4. **Viral Load Scaling**:
   - Peak load = 100% (1.0)
   - All values relative to maximum
   - Example: 0.5 = 50% of peak viral load

5. **Intervention Scaling**:
   - Rates represent daily percentage of population
   - Example: 5% quarantine = 50,000 people per day
   - Rates can exceed 100% for multiple interventions

6. **Output Scaling**:
   - All results converted to percentages
   - Example: 0.02 = 2% of population
   - Easy conversion to absolute numbers

### Practical Examples

1. **Outbreak Size**:
   - 5% total infected = 50,000 people
   - 0.1% daily new cases = 1,000 people
   - 0.02% mortality = 200 deaths

2. **Intervention Impact**:
   - 40% quarantine = 400,000 isolated
   - 10% vaccination = 100,000 protected
   - Combined effect reduces transmission by 50%

3. **Healthcare Demand**:
   - 1% hospitalized = 10,000 beds
   - 0.1% ICU = 1,000 critical cases
   - Peak demand shows maximum strain

4. **Economic Impact**:
   - 2% workforce infected = 20,000 workers
   - 0.5% daily absenteeism = 5,000 people
   - Productivity loss as percentage of total

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
- All metrics are presented as percentages of the total population

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
   - Total population of 1 million people

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