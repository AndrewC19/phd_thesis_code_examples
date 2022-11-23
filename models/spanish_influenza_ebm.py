from scipy.integrate import odeint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{libertine}"
    }
)

# Model parameters (at point of initialisation)
START_MORTALITY_PROB = 0.01
START_RECOVERY_TIME = 2.5
START_MORTALITY_TIME = 1
START_TRANSMISSION_PROB = 0.15
START_ENCOUNTER_RATE = 4
START_INCUBATION_TIME = 3

# Initial compartments
START_REMOVED = 0
START_DECEASED = 0
START_RECOVERED = 0
START_INFECTIOUS = 1000
START_INCUBATING = 0
START_SUSCEPTIBLE = 103267000
START_TOTAL_INFECTED = 0

# Time settings
DURATION = 200
RESOLUTION = DURATION * 1


class SpanishInfluenzaEBM:

    def __init__(self):
        # Disease transmission parameters
        self.mortality_prob: float = START_MORTALITY_PROB
        self.recovery_time: float = START_RECOVERY_TIME
        self.mortality_time: float = START_MORTALITY_TIME
        self.transmission_prob: float = START_TRANSMISSION_PROB
        self.encounter_rate: float = START_ENCOUNTER_RATE
        self.incubation_time: float = START_INCUBATION_TIME

        # Duration to simulate
        self.time_points: list = np.linspace(0, DURATION, RESOLUTION)

    def compute_population(self, recovered, infectious, incubating, susceptible):
        return recovered + infectious + incubating + susceptible

    def compute_r(self, susceptible, population):
        return (self.transmission_prob * self.encounter_rate *
                susceptible) / population

    def update_deceased(self, removed):
        return self.mortality_prob * removed

    def update_recovered(self, removed):
        return (1 - self.mortality_prob) * removed

    def compute_time_step(self, y, t):
        recovered, infectious, susceptible, incubating, total_infected = y

        # Compute constants
        population = self.compute_population(recovered, infectious,
                                             incubating, susceptible)
        r = self.compute_r(susceptible, population)

        # Compute time derivatives
        d_removed_dt = (((self.mortality_prob / self.mortality_time) +
                         (1 - self.mortality_prob) / self.recovery_time) *
                        infectious)
        d_infectious_dt = -d_removed_dt + (incubating /
                                           self.incubation_time)
        d_susceptible_dt = -r * infectious
        d_incubating_dt = ((-incubating / self.incubation_time) +
                           r * infectious)
        d_total_infected_dt = incubating / self.incubation_time
        return (d_removed_dt, d_infectious_dt, d_susceptible_dt,
                d_incubating_dt, d_total_infected_dt)

    def solve(self,
              initial_recovered=START_RECOVERED,
              initial_infectious=START_INFECTIOUS,
              initial_susceptible=START_SUSCEPTIBLE,
              initial_incubating=START_INCUBATING,
              initial_total_infected=START_TOTAL_INFECTED):
        # Define the initial states
        y0 = (initial_recovered, initial_infectious, initial_susceptible,
              initial_incubating,initial_total_infected)

        # Solve ODE starting from initial states
        y = odeint(self.compute_time_step,
                   y0,
                   self.time_points)

        # Store results in a dataframe
        results = pd.DataFrame({'recovered': y[:, 0],
                                'infectious': y[:, 1],
                                'susceptible': y[:, 2],
                                'incubating': y[:, 3],
                                'total_infected': y[:, 4]})
        results.index.names = ['t']
        return results

    @staticmethod
    def plot(results_df):
        xs = results_df.index
        for col in results_df.columns:
            plt.plot(xs, results_df[col], color='red', linewidth=1)
            title_str = col.replace('_', ' ')
            title_case_title_str = title_str.title()
            plt.xlabel("Days")
            plt.ylabel(title_case_title_str)
            plt.show()


if __name__ == "__main__":
    spanish_influenza_model = SpanishInfluenzaEBM()
    spanish_influenza_df = spanish_influenza_model.solve()
    print(spanish_influenza_df)
    spanish_influenza_model.plot(spanish_influenza_df)
