from dataclasses import dataclass
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

@dataclass
class SpanishInfluenzaEBM:
    # Population-level outputs
    population: float
    r: float

    # Disease progression states
    removed: float = START_REMOVED,
    deceased: float = START_DECEASED,
    recovered: float = START_RECOVERED,
    infectious: float = START_INFECTIOUS,
    incubating: float = START_INCUBATING,
    susceptible: float = START_SUSCEPTIBLE,
    total_infected: float = START_TOTAL_INFECTED

    # Disease transmission parameters
    mortality_prob: float = START_MORTALITY_PROB,
    recovery_time: float = START_RECOVERY_TIME,
    mortality_time: float = START_MORTALITY_TIME,
    transmission_prob: float = START_TRANSMISSION_PROB,
    encounter_rate: float = START_ENCOUNTER_RATE,
    incubation_time: float = START_INCUBATION_TIME,


    def update_population(self):
        self.population += (self.recovered + self.infectious +
                            self.incubating + self.susceptible)

    def update_r(self):
        self.r += ((self.transmission_prob * self.encounter_rate *
                   self.susceptible) / self.population)

    def update_recovered(self):
        self.recovered += ((1 - self.mortality_prob) * self.removed)



if __name__ == "__main__":
    print("Main")
