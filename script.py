"""
Age-Structured SIRQV Model with Viral Load and Mortality Dynamics

This script implements an epidemiological model that combines:
1. Age-structured population with different epidemiological characteristics
2. Viral load dynamics affecting transmission rates
3. Quarantine and vaccination interventions
4. Age-specific mortality tracking

The model is particularly relevant for studying COVID-19 dynamics, capturing:
- Age-specific susceptibility and severity
- Time-dependent infectiousness through viral load
- Realistic contact patterns between age groups
- Public health interventions
- Mortality patterns and risk factors

For detailed model assumptions, see model_assumptions.txt
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# Set plotting style
plt.style.use('default')
sns.set_theme()
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Create main output directory
os.makedirs("output", exist_ok=True)

@dataclass
class SimulationParameters:
    """Container for simulation parameters"""
    quarantine_rate: float
    vaccination_rate: float
    description: str

def create_simulation_scenarios() -> List[SimulationParameters]:
    """Create different simulation scenarios"""
    return [
        SimulationParameters(0.05, 0.01, "baseline"),
        SimulationParameters(0.40, 0.01, "high_quarantine"),  # Very high quarantine
        SimulationParameters(0.05, 0.10, "high_vaccination"),  # Very high vaccination
        SimulationParameters(0.40, 0.10, "high_both"),         # Both very high
        SimulationParameters(0.01, 0.001, "low_both"),        # Both very low
    ]

@dataclass
class AgeGroup:
    """Container for age group specific parameters"""
    name: str
    susceptibility: float        # Relative susceptibility to infection
    recovery_rate: float        # Relative recovery rate
    vaccine_effectiveness: float # Vaccine effectiveness
    population_fraction: float  # Fraction of total population
    mortality_rate: float       # Base mortality rate
    viral_load_sensitivity: float # How much viral load affects mortality

class AgeStructuredViralLoadModel:
    """
    Implements an age-structured SIRQVD model with viral load and mortality dynamics.
    
    The model divides the population into age groups and tracks:
    - Susceptible (S)
    - Infected (I)
    - Quarantined (Q)
    - Recovered (R)
    - Vaccinated (V)
    - Dead (D)
    - Viral Load (L)
    
    for each age group.
    """
    
    def __init__(self, age_groups: List[AgeGroup], contact_matrix: np.ndarray, params: SimulationParameters):
        """Initialize the model with age groups and contact patterns"""
        self.age_groups = age_groups
        self.contact_matrix = contact_matrix
        self.n_age_groups = len(age_groups)
        self.params = params
        
        # Viral load parameters
        self.incubation_days = 3
        self.peak_days = 5
        self.decline_days = 7
        self.max_viral_load = 1000
        
        # Base epidemiological parameters
        self.base_beta = 0.3
        self.base_gamma = 0.1
        self.theta = params.quarantine_rate  # Quarantine rate
        self.nu = params.vaccination_rate   # Vaccination rate

    def viral_load_curve(self, t: float) -> float:
        """Calculate viral load at time t since infection"""
        if t < self.incubation_days:
            return 0.1 * np.exp(t/2)
        elif t < self.incubation_days + self.peak_days:
            return 1.0
        else:
            return np.exp(-(t - (self.incubation_days + self.peak_days)) / self.decline_days)

    def mortality_rate(self, age_index: int, viral_load: float) -> float:
        """Calculate mortality rate based on age and viral load"""
        group = self.age_groups[age_index]
        # Mortality increases with viral load and age-specific sensitivity
        return group.mortality_rate * (1 + group.viral_load_sensitivity * viral_load)

    def effective_beta(self, age_i: int, age_j: int, viral_load: float) -> float:
        """Calculate effective transmission rate between age groups"""
        return (self.base_beta * 
                self.contact_matrix[age_i, age_j] * 
                self.age_groups[age_i].susceptibility * 
                viral_load)

    def deriv(self, state: np.ndarray, t: float) -> np.ndarray:
        """Calculate derivatives for the age-structured model"""
        n_vars = 7  # S, I, Q, R, V, D, L for each age group
        derivatives = np.zeros(self.n_age_groups * n_vars)
        
        # Reshape state into age groups
        state_matrix = state.reshape(self.n_age_groups, n_vars)
        
        for i in range(self.n_age_groups):
            # Unpack state for age group i
            S_i, I_i, Q_i, R_i, V_i, D_i, L_i = state_matrix[i]
            
            # Calculate total population
            N = np.sum(state_matrix[:, :5])  # S + I + Q + R + V
            
            # Calculate new infections
            new_infections = 0
            for j in range(self.n_age_groups):
                if I_i > 0:
                    beta_ij = self.effective_beta(j, i, L_i)
                    new_infections += beta_ij * S_i * state_matrix[j, 1] / N
            
            # Calculate mortality rate
            mu = self.mortality_rate(i, L_i)
            
            # Calculate derivatives
            dSdt = -new_infections - self.nu * S_i * self.age_groups[i].vaccine_effectiveness
            dIdt = new_infections - self.gamma_i(i) * I_i - self.theta * I_i - mu * I_i
            dQdt = self.theta * I_i - self.gamma_i(i) * Q_i - mu * Q_i
            dRdt = self.gamma_i(i) * (I_i + Q_i)
            dVdt = self.nu * S_i * self.age_groups[i].vaccine_effectiveness
            dDdt = mu * (I_i + Q_i)  # Deaths from both infected and quarantined
            
            # Viral load dynamics
            if I_i > 0:
                dLdt = self.viral_load_curve(t) - 0.1 * L_i
            else:
                dLdt = 0
            
            # Store derivatives
            derivatives[i*n_vars:(i+1)*n_vars] = [dSdt, dIdt, dQdt, dRdt, dVdt, dDdt, dLdt]
        
        return derivatives

    def gamma_i(self, age_index: int) -> float:
        """Recovery rate for age group i"""
        return self.base_gamma * self.age_groups[age_index].recovery_rate

    def simulate(self, t: np.ndarray, initial_conditions: np.ndarray) -> np.ndarray:
        """Run simulation over time"""
        return odeint(self.deriv, initial_conditions, t)

def create_age_groups() -> List[AgeGroup]:
    """Define age groups and their parameters"""
    return [
        # name, susceptibility, recovery_rate, vaccine_effectiveness, population_fraction, mortality_rate, viral_load_sensitivity
        AgeGroup("0-19", 0.8, 1.2, 0.7, 0.25, 0.0002, 1.2),    # Children/Teens - slightly higher mortality
        AgeGroup("20-39", 1.0, 1.0, 0.8, 0.30, 0.001, 1.8),    # Young Adults - higher mortality
        AgeGroup("40-59", 1.2, 0.9, 0.85, 0.25, 0.005, 2.5),   # Middle Age - significantly higher mortality
        AgeGroup("60+", 1.5, 0.7, 0.9, 0.20, 0.02, 3.5)        # Elderly - much higher mortality
    ]

def create_contact_matrix(n_groups: int) -> np.ndarray:
    """Create a contact matrix between age groups"""
    matrix = np.array([
        [3.0, 1.5, 0.5, 0.2],  # Children/Teens
        [1.5, 2.0, 1.0, 0.3],  # Young Adults
        [0.5, 1.0, 1.5, 0.5],  # Middle Age
        [0.2, 0.3, 0.5, 1.0]   # Elderly
    ])
    return matrix

def get_age_midpoint(age_group: str) -> float:
    """Calculate the midpoint of an age group"""
    if '+' in age_group:
        return 70  # Assuming 60+ means average age of 70
    ages = age_group.split('-')
    return (int(ages[0]) + int(ages[1])) / 2

def analyze_mortality(results: np.ndarray, t: np.ndarray, age_groups: List[AgeGroup], output_dir: str, params: SimulationParameters):
    """Analyze and plot mortality patterns"""
    n_groups = len(age_groups)
    n_vars = 7  # S, I, Q, R, V, D, L
    
    # Create analysis directory
    analysis_dir = f"{output_dir}/analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save simulation parameters
    with open(f"{analysis_dir}/parameters.json", "w") as f:
        json.dump({
            "quarantine_rate": params.quarantine_rate,
            "vaccination_rate": params.vaccination_rate,
            "description": params.description
        }, f, indent=4)
    
    # 1. Deaths by age group over time
    plt.figure(figsize=(12, 8))
    for i in range(n_groups):
        deaths = results[:, i*n_vars + 5]  # D compartment
        plt.plot(t, deaths, label=age_groups[i].name)
    plt.title('Cumulative Deaths by Age Group')
    plt.xlabel('Time (days)')
    plt.ylabel('Deaths')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{analysis_dir}/deaths_by_age.png")
    plt.close()
    
    # 2. Death rates vs viral load
    plt.figure(figsize=(12, 8))
    for i in range(n_groups):
        viral_load = results[:, i*n_vars + 6]  # L compartment
        deaths = np.diff(results[:, i*n_vars + 5], prepend=0)  # Daily deaths
        plt.scatter(viral_load, deaths, label=age_groups[i].name, alpha=0.5)
    plt.title('Daily Deaths vs Viral Load')
    plt.xlabel('Viral Load')
    plt.ylabel('Daily Deaths')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{analysis_dir}/deaths_vs_viral_load.png")
    plt.close()
    
    # 3. Time to death analysis
    plt.figure(figsize=(12, 8))
    for i in range(n_groups):
        daily_deaths = np.diff(results[:, i*n_vars + 5], prepend=0)
        plt.plot(t, daily_deaths, label=age_groups[i].name)
    plt.title('Daily Deaths Over Time')
    plt.xlabel('Time (days)')
    plt.ylabel('Daily Deaths')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{analysis_dir}/daily_deaths.png")
    plt.close()
    
    # 4. Mortality risk factors
    plt.figure(figsize=(12, 8))
    final_deaths = results[-1, 5::n_vars]  # Final death counts
    age_names = [group.name for group in age_groups]
    plt.bar(age_names, final_deaths)
    plt.title('Total Deaths by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Total Deaths')
    plt.grid(True)
    plt.savefig(f"{analysis_dir}/total_deaths.png")
    plt.close()
    
    # 5. Save detailed analysis results with enhanced metrics
    analysis_results = {
        "age_groups": {},
        "summary": {
            "total_deaths": float(np.sum(results[-1, 5::n_vars])),
            "peak_daily_deaths": float(np.max(np.diff(results[:, 5::n_vars], axis=0))),
            "average_viral_load": float(np.mean(results[:, 6::n_vars])),
            "total_infected": float(np.sum(results[-1, 1::n_vars])),  # Total infected at end
            "peak_infected": float(np.max(results[:, 1::n_vars])),    # Peak infected count
            "total_recovered": float(np.sum(results[-1, 3::n_vars])), # Total recovered
            "total_vaccinated": float(np.sum(results[-1, 4::n_vars])),# Total vaccinated
            "infection_fatality_ratio": float(np.sum(results[-1, 5::n_vars]) / np.sum(results[-1, 1::n_vars]) * 100),
            "time_to_peak_deaths": float(t[np.argmax(np.diff(results[:, 5::n_vars], axis=0))]),
            "average_time_to_death": float(np.mean([t[np.argmax(np.diff(results[:, i*n_vars + 5], prepend=0))] for i in range(n_groups)])),
            "mortality_rate_per_100k": float(np.sum(results[-1, 5::n_vars]) * 100000)
        }
    }
    
    for i, group in enumerate(age_groups):
        total_deaths = results[-1, i*n_vars + 5]
        peak_deaths = np.max(np.diff(results[:, i*n_vars + 5], prepend=0))
        avg_viral_load = np.mean(results[:, i*n_vars + 6])
        total_infected = results[-1, i*n_vars + 1] + results[-1, i*n_vars + 2]  # I + Q
        peak_infected = np.max(results[:, i*n_vars + 1] + results[:, i*n_vars + 2])
        age_midpoint = get_age_midpoint(group.name)
        
        analysis_results["age_groups"][group.name] = {
            "total_deaths": float(total_deaths),
            "peak_daily_deaths": float(peak_deaths),
            "average_viral_load": float(avg_viral_load),
            "case_fatality_rate": float(total_deaths / group.population_fraction * 100),
            "mortality_rate": float(group.mortality_rate),
            "viral_load_sensitivity": float(group.viral_load_sensitivity),
            "total_infected": float(total_infected),
            "peak_infected": float(peak_infected),
            "infection_fatality_ratio": float(total_deaths / total_infected * 100 if total_infected > 0 else 0),
            "time_to_peak_deaths": float(t[np.argmax(np.diff(results[:, i*n_vars + 5], prepend=0))]),
            "mortality_rate_per_100k": float(total_deaths * 100000),
            "years_of_life_lost": float(total_deaths * (80 - age_midpoint))  # Rough estimate
        }
    
    with open(f"{analysis_dir}/detailed_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=4)

def compare_scenarios(scenario_dirs: List[str], output_dir: str):
    """Create comparison visualizations across all scenarios"""
    comparison_dir = f"{output_dir}/comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Load all scenario data
    scenario_data = []
    for dir_path in scenario_dirs:
        with open(f"{dir_path}/analysis/parameters.json", "r") as f:
            params = json.load(f)
        with open(f"{dir_path}/analysis/detailed_analysis.json", "r") as f:
            analysis = json.load(f)
        scenario_data.append((params, analysis))
    
    # 1. Compare total deaths across scenarios
    plt.figure(figsize=(15, 8))
    scenarios = [data[0]["description"] for data in scenario_data]
    total_deaths = [data[1]["summary"]["total_deaths"] for data in scenario_data]
    plt.bar(scenarios, total_deaths)
    plt.title('Total Deaths by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Total Deaths')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/total_deaths_comparison.png")
    plt.close()
    
    # 2. Compare peak daily deaths
    plt.figure(figsize=(15, 8))
    peak_deaths = [data[1]["summary"]["peak_daily_deaths"] for data in scenario_data]
    plt.bar(scenarios, peak_deaths)
    plt.title('Peak Daily Deaths by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Peak Daily Deaths')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/peak_deaths_comparison.png")
    plt.close()
    
    # 3. Compare infection fatality ratios
    plt.figure(figsize=(15, 8))
    ifr = [data[1]["summary"]["infection_fatality_ratio"] for data in scenario_data]
    plt.bar(scenarios, ifr)
    plt.title('Infection Fatality Ratio by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('IFR (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/ifr_comparison.png")
    plt.close()
    
    # 4. Compare age-specific mortality rates
    age_groups = list(scenario_data[0][1]["age_groups"].keys())
    plt.figure(figsize=(15, 8))
    width = 0.15
    x = np.arange(len(age_groups))
    
    for i, (params, analysis) in enumerate(scenario_data):
        mortality_rates = [analysis["age_groups"][age]["mortality_rate_per_100k"] for age in age_groups]
        plt.bar(x + i*width, mortality_rates, width, label=params["description"])
    
    plt.title('Age-Specific Mortality Rates by Scenario')
    plt.xlabel('Age Group')
    plt.ylabel('Mortality Rate per 100,000')
    plt.xticks(x + width*2, age_groups)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/age_specific_mortality_comparison.png")
    plt.close()
    
    # 5. Compare years of life lost
    plt.figure(figsize=(15, 8))
    yll = [sum(data[1]["age_groups"][age]["years_of_life_lost"] for age in age_groups) for data in scenario_data]
    plt.bar(scenarios, yll)
    plt.title('Years of Life Lost by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Years of Life Lost')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/yll_comparison.png")
    plt.close()
    
    # 6. Save comparison summary
    comparison_summary = {
        "scenarios": [data[0]["description"] for data in scenario_data],
        "summary_metrics": {
            "total_deaths": [data[1]["summary"]["total_deaths"] for data in scenario_data],
            "peak_daily_deaths": [data[1]["summary"]["peak_daily_deaths"] for data in scenario_data],
            "infection_fatality_ratio": [data[1]["summary"]["infection_fatality_ratio"] for data in scenario_data],
            "years_of_life_lost": [sum(data[1]["age_groups"][age]["years_of_life_lost"] for age in age_groups) for data in scenario_data]
        }
    }
    
    with open(f"{comparison_dir}/comparison_summary.json", "w") as f:
        json.dump(comparison_summary, f, indent=4)

def run_simulation(params: SimulationParameters) -> str:
    """Run a single simulation with given parameters"""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/simulation_{params.description}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Time points for simulation
    t = np.linspace(0, 180, 1000)
    
    # Create age groups and contact matrix
    age_groups = create_age_groups()
    contact_matrix = create_contact_matrix(len(age_groups))
    
    # Create model
    model = AgeStructuredViralLoadModel(age_groups, contact_matrix, params)
    
    # Initial conditions
    initial_state = np.zeros(len(age_groups) * 7)  # 7 variables per age group
    for i, group in enumerate(age_groups):
        # Set initial susceptible population
        initial_state[i*7] = group.population_fraction * 0.99
        # Set initial infected population
        initial_state[i*7 + 1] = group.population_fraction * 0.01
        # Initial viral load
        initial_state[i*7 + 6] = 0.1
    
    # Run simulation
    results = model.simulate(t, initial_state)
    
    # Analyze mortality patterns
    analyze_mortality(results, t, age_groups, output_dir, params)
    
    return output_dir

def main():
    """Run multiple simulations with different parameters"""
    scenarios = create_simulation_scenarios()
    scenario_dirs = []
    
    print("Starting multiple simulations...")
    for params in scenarios:
        output_dir = run_simulation(params)
        scenario_dirs.append(output_dir)
        print(f"Completed simulation: {params.description}")
        print(f"Results saved to: {output_dir}")
    
    # Create comparison visualizations
    print("\nCreating comparison visualizations...")
    compare_scenarios(scenario_dirs, "output")
    print("Comparison visualizations complete!")
    
    print("\nAll simulations and analysis completed!")

if __name__ == "__main__":
    main()
