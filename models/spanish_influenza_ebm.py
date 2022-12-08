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

    def __init__(self,
                 mortality_prob: float = START_MORTALITY_PROB,
                 recovery_time: float = START_RECOVERY_TIME,
                 mortality_time: float = START_MORTALITY_TIME,
                 transmission_prob: float = START_TRANSMISSION_PROB,
                 encounter_rate: float = START_ENCOUNTER_RATE,
                 incubation_time: float = START_INCUBATION_TIME
                 ):

        # Disease transmission parameters
        self.mortality_prob = mortality_prob
        self.recovery_time = recovery_time
        self.mortality_time = mortality_time
        self.transmission_prob = transmission_prob
        self.encounter_rate = encounter_rate
        self.incubation_time = incubation_time

        # Duration to simulate
        self.time_points: list = np.linspace(0, DURATION, RESOLUTION)

        # Population size
        self.population_size = None

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
        (total_infected, susceptible,
         incubating, infected,
         dead, recovered) = states

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
        updates_states = (d_total_infected_dt, d_susceptible_dt,
                          d_incubating_dt, d_infectious_dt,
                          d_deceased_dt, d_recovered_dt)
        return updates_states

    def solve(self,
              initial_susceptible=START_SUSCEPTIBLE,
              initial_incubating=START_INCUBATING,
              initial_infectious=START_INFECTIOUS,
              initial_recovered=START_RECOVERED,
              initial_deceased=START_DECEASED,
              initial_total_infected=START_TOTAL_INFECTED,
              plot: bool = False,
              out_path: str = None):
        self.population_size = self.compute_population(initial_susceptible,
                                                       initial_incubating,
                                                       initial_infectious,
                                                       initial_recovered)

        # Define the initial states
        y0 = (initial_total_infected, initial_susceptible,
              initial_incubating, initial_infectious,
              initial_deceased, initial_recovered)

        # Solve ODE starting from initial states
        y = odeint(self.model,
                   y0,
                   t=self.time_points)

        # Store results in a dataframe
        results = pd.DataFrame({'total_infected': y[:, 0],
                                'susceptible': y[:, 1],
                                'incubating': y[:, 2],
                                'infectious': y[:, 3],
                                'deceased': y[:, 4],
                                'recovered': y[:, 5]})
        results.index.names = ['t']

        # Plot results
        if plot:
            self.plot(results, out_path)
        return results

    @staticmethod
    def plot(results_df, out_path):
        fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 7))
        xs = results_df.index

        # Make plots resemble those from Sukumar and Nutaro's paper
        ax[0, 0].plot(xs, results_df["total_infected"], color='red',
                      linewidth=1)
        ax[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
        ax[0, 0].set_ylim(0, 6e7)
        ax[0, 0].set_ylabel("Total Infected")

        ax[0, 1].plot(xs, results_df["susceptible"], color='red',
                      linewidth=1)
        ax[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(1, 2))
        ax[0, 1].set_ylim(4e7, 12e7)
        ax[0, 1].set_ylabel("Susceptible")

        ax[1, 0].plot(xs, results_df["incubating"], color='red',
                      linewidth=1)
        ax[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
        ax[1, 0].set_ylim(0, 6e6)
        ax[1, 0].set_ylabel("Incubating")

        ax[1, 1].plot(xs, results_df["infectious"], color='red',
                      linewidth=1)
        ax[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
        ax[1, 1].set_ylim(0, 4e6)
        ax[1, 1].set_ylabel("Infectious")

        ax[2, 0].plot(xs, results_df["deceased"], color='red',
                      linewidth=1)
        ax[2, 0].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
        ax[2, 0].set_ylim(0, 6e5)
        ax[2, 0].set_ylabel("Deceased")
        ax[2, 0].set_xlabel("Days")

        ax[2, 1].plot(xs, results_df["recovered"], color='red',
                      linewidth=1)
        ax[2, 1].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
        ax[2, 1].set_ylim(0, 6e7)
        ax[2, 1].set_ylabel("Recovered")
        ax[2, 1].set_xlabel("Days")
        plt.suptitle("Spanish Influenza 1918 in USA")
        plt.tight_layout()
        plt.show()
        if out_path:
            fig.savefig(out_path, format='pdf', dpi=300)


if __name__ == "__main__":
    spanish_influenza_model = SpanishInfluenzaEBM()
    spanish_influenza_df = spanish_influenza_model.solve(
        out_path="spanish_influenza_ebm.pdf")
