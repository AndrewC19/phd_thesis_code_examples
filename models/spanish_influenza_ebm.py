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
START_MORTALITY_PROB = 0.01  # GAMMA
START_RECOVERY_TIME = 2.5
START_MORTALITY_TIME = 1
START_TRANSMISSION_PROB = 0.15  # BETA
START_ENCOUNTER_RATE = 4
START_INCUBATION_TIME = 3  # 1/SIGMA

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

    @staticmethod
    def compute_population(susceptible, incubating, infected, recovered):
        """Equation for calculating the current living population.

        This equation computes the total living population, given the
        distribution of susceptible, incubating, infected, and recovered
        individuals."""
        return susceptible + incubating + infected + recovered

    def compute_r(self, susceptible, population):
        """Equation for calculating the current R value in the population.

        This equation computes the current R value, given the number of
        susceptible individuals and the total number of individuals in the
        population."""
        return (self.transmission_prob * self.encounter_rate *
                susceptible) / population

    def dDdt(self, to_remove):
        """Equation for the derivative of deceased.

        This equation computes the change in deceased per time step, given the
        number of individuals to remove in this time step and the probability of
         mortality."""
        return self.mortality_prob * to_remove

    def dRdt(self, to_remove):
        """Equation for the derivative of recovered.

        This equation computes the change in recovered per time step, given the
        number of individuals to remove in this time step and the probability of
         mortality."""
        return (1 - self.mortality_prob) * to_remove

    def dRemoved_dt(self, infected):
        """Equation for the derivative of the number of removed individuals.

        This equation computes the number of individuals to remove per time
        step, given the number of infected individuals, the probability of
        mortality, time for mortality, and time for recovery."""
        return ((self.mortality_prob / self.mortality_time +
                 (1 - self.mortality_prob) / self.recovery_time) *
                infected)

    def dInfectious_dt(self, d_removed_dt, incubating):
        """Equation for the derivative of the number of infectious individuals.

        This equation computes change in infected individuals per time step,
        given the number of infected and incubating individuals, and the
        incubation time."""
        return -d_removed_dt + incubating / self.incubation_time

    @staticmethod
    def dSdt(r, infected):
        """Equation for the derivative of the number of susceptible individuals.

        This equation computes the change in the number of susceptible
        individuals per time step, given the current R number and infected
        individuals."""
        return -r * infected

    def dIncubating_dt(self, incubating, r, infected):
        """Equation for the derivative of the number of incubating individuals.

        This equation computes the change in the number of incubating
        individuals per time step, given the current number of incubating and
        infected individuals, incubation time, and R number."""
        return -incubating / self.incubation_time + r * infected

    def dTotalInfected_df(self, incubating):
        """Equation for the derivative of the total infected individuals.

        This equation computes the change in the total number of infected
        individuals per time step, given the current number of incubating
        individuals and the incubation time."""
        return incubating / self.incubation_time

    def model(self, states: tuple, t):
        (susceptible, incubating, infected, recovered,
         dead, total_infected) = states

        # Compute constants
        population = self.compute_population(susceptible, incubating,
                                             infected, recovered)
        r = self.compute_r(susceptible, population)

        # Compute derivatives
        d_removed_dt = self.dRemoved_dt(infected)
        d_deceased_dt = self.dDdt(d_removed_dt)
        d_recovered_dt = self.dRdt(d_removed_dt)
        d_infectious_dt = self.dInfectious_dt(d_removed_dt, incubating)
        d_susceptible_dt = self.dSdt(r, infected)
        d_incubating_dt = self.dIncubating_dt(incubating, r, infected)
        d_total_infected_dt = self.dTotalInfected_df(incubating)

        # Update the states
        updates_states = (d_susceptible_dt, d_incubating_dt, d_infectious_dt,
                          d_recovered_dt, d_deceased_dt, d_total_infected_dt)
        return updates_states

    def solve(self,
              initial_susceptible=START_SUSCEPTIBLE,
              initial_incubating=START_INCUBATING,
              initial_infectious=START_INFECTIOUS,
              initial_recovered=START_RECOVERED,
              initial_deceased=START_DECEASED,
              initial_total_infected=START_TOTAL_INFECTED,
              plot: bool = True):

        # Define the initial states
        y0 = (initial_susceptible, initial_incubating, initial_infectious,
              initial_recovered, initial_deceased, initial_total_infected)

        # Solve ODE starting from initial states
        y = odeint(self.model,
                   y0,
                   t=self.time_points)

        # Store results in a dataframe
        results = pd.DataFrame({'susceptible': y[:, 0],
                                'incubating': y[:, 1],
                                'infectious': y[:, 2],
                                'recovered': y[:, 3],
                                'deceased': y[:, 4],
                                'total_infected': y[:, 5]})
        results.index.names = ['t']

        # Plot results
        if plot:
            self.plot(results)
        return results

    @staticmethod
    def plot(results_df):
        xs = results_df.index
        for col in results_df.columns:
            plt.plot(xs, results_df[col], color='red', linewidth=1)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
            title_str = col.replace('_', ' ')
            title_case_title_str = title_str.title()
            plt.xlabel("Days")
            plt.ylabel(title_case_title_str)
            plt.show()


if __name__ == "__main__":
    spanish_influenza_model = SpanishInfluenzaEBM()
    spanish_influenza_df = spanish_influenza_model.solve()
