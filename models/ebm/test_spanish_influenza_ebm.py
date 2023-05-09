"""
Metamorphic relations for the Spanish Influenza model as presented by Pullum
and Ozmen in their 2012 paper entitled: Early Results from Metamorphic Testing
of Epidemiological Models.
"""

import numpy as np
import pandas as pd
import random

from spanish_influenza_ebm import SpanishInfluenzaEBM
from itertools import product
import time

OUTPUT_CSV_PATH = "metamorphic_testing_data.csv"

random.seed(123)  # set seed for reproducibility


def random_mortality_prob():
    return np.random.uniform(0.0025, 0.04)


def random_recovery_time():
    return np.random.uniform(0.625, 10)


def random_mortality_time():
    return np.random.uniform(0.25, 4)


def random_transmission_prob():
    return np.random.uniform(0.0375, 0.6)


def random_encounter_rate():
    return np.random.uniform(1, 16)


def random_incubation_time():
    return np.random.uniform(0.75, 12)


def random_test_case():
    return {"mortality_prob": random_mortality_prob(),
            "recovery_time": random_recovery_time(),
            "mortality_time": random_mortality_time(),
            "transmission_prob": random_transmission_prob(),
            "encounter_rate": random_encounter_rate(),
            "incubation_time": random_incubation_time()}


def setup_module():
    inputs = ["mortality_prob", "recovery_time", "mortality_time",
              "transmission_prob", "encounter_rate", "incubation_time"]
    outputs = ["total_infected", "susceptible", "incubating",
               "infectious", "deceased", "recovered"]
    simulation_days = list(range(0, 200))
    daily_outputs = [f"{o}_{n}" for (o, n) in product(outputs, simulation_days)]
    header = inputs + daily_outputs + ["test_pass"]
    empty_df = pd.DataFrame(columns=header)
    empty_df.to_csv(OUTPUT_CSV_PATH)
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)
    end_time = time.time()
    ex_time = end_time - start_time
    append_results_to_df(source_model, source_model_df, pd.NA, "source",
                         ex_time)


def append_results_to_df(model, results_df, test_pass, relation, ex_time):
    full_results_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])
    results_dict = {}

    # Convert outputs to row form
    for col in results_df:
        for time_step, val in enumerate(results_df[col]):
            full_results_col = f"{col}_{time_step}"
            results_dict[full_results_col] = val

    # Get value of each input from model
    results_dict["mortality_prob"] = model.mortality_prob
    results_dict["recovery_time"] = model.recovery_time
    results_dict["mortality_time"] = model.mortality_time
    results_dict["transmission_prob"] = model.transmission_prob
    results_dict["encounter_rate"] = model.encounter_rate
    results_dict["incubation_time"] = model.incubation_time
    results_dict["test_pass"] = test_pass
    results_dict["relation"] = relation
    results_dict["time"] = ex_time

    # Append inputs and outputs to existing df
    results_to_append_series = pd.Series(results_dict)
    full_results_df = full_results_df.append(results_to_append_series,
                                             ignore_index=True)
    full_results_df.to_csv(OUTPUT_CSV_PATH)


def test_MR1():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(0, 0.999)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.mortality_prob *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Check 1) (Deceased decreases by less than a factor of n)
    deceased_delta = (source_model_df["deceased"].iloc[-1] -
                      follow_up_model_df["deceased"].iloc[-1])

    # Does this cause a decrease?
    test_a = 0 < deceased_delta

    # Is the magnitude of the effect a factor of at most n?
    test_b = abs(source_model.population_size*n) > abs(deceased_delta)

    # Check 2) (Recovered increases by more than 1/n)
    # NOTE: error in Table 1 of the original paper; row 2 of MR1 should say
    # increases the number of recovered by more than 1/n, not decrease. It
    # doesn't make sense for a reduction in mortality probability to cause a
    # decrease in recoveries.
    recovered_delta = (source_model_df["recovered"].iloc[-1] -
                       follow_up_model_df["recovered"].iloc[-1])

    # Does this cause an increase?
    test_c = 0 > recovered_delta

    # Is the magnitude of the effect greater than 1/n?
    test_d = abs(1/n) < abs(recovered_delta)

    test_pass = test_a and test_b and test_c and test_d
    end_time = time.time()
    execution_time = end_time - start_time
    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR1",
                         execution_time)
    assert test_a
    assert test_b
    assert test_c
    assert test_d


def test_MR2():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(1.001, 30)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.mortality_prob *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Check 1) (Deceased increases by less than a factor of n)
    deceased_delta = (source_model_df["deceased"].iloc[-1] -
                      follow_up_model_df["deceased"].iloc[-1])

    # Does this cause an increase?
    test_a = 0 > deceased_delta

    # Is the magnitude of the effect a factor less than n?
    test_b = abs(source_model.population_size*n) > abs(deceased_delta)

    # Check 2) (Recovered increases by more than 1/n)
    # NOTE: error in Table 1 of the original paper; row 2 of MR2 should say
    # decreases the number recovered by more than 1/n, not increases. It
    # doesn't make sense for an increase in mortality probability to cause an
    # increase in recoveries.
    recovered_delta = (source_model_df["recovered"].iloc[-1] -
                       follow_up_model_df["recovered"].iloc[-1])

    # Does this cause a decrease?
    test_c = 0 < recovered_delta

    # Is the magnitude of the effect greater than t 1/n?
    test_d = abs(1 / n) < abs(recovered_delta)

    test_pass = test_a and test_b and test_c and test_d

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR2",
                         execution_time)
    assert test_a
    assert test_b
    assert test_c
    assert test_d


def test_MR3():
    # START_INFECTIOUS = 1000
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(0, 0.999)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model_df = follow_up_model.solve(plot=False,
                                               out_path="spanish_influenza_ebm_small_change_in_I0.pdf",
                                               initial_infectious=1000*n)

    # Approximation: The height of the pandemic deaths should be delayed
    source_deceased_gradient = np.gradient(source_model_df["deceased"])
    source_deceased_max_gradient_day = np.argmax(source_deceased_gradient)

    follow_up_deceased_gradient = np.gradient(follow_up_model_df["deceased"])
    follow_up_deceased_max_gradient_day = np.argmax(follow_up_deceased_gradient)

    test_a = (follow_up_deceased_max_gradient_day >
              source_deceased_max_gradient_day)

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_a, "MR3",
                         execution_time)

    assert test_a


def test_MR4():
    # START_INFECTIOUS = 1000
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(1.001, 30)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model_df = follow_up_model.solve(plot=False,
                                               initial_infectious=1000*n)

    # Approximation: The height of the pandemic deaths should be earlier
    source_deceased_gradient = np.gradient(source_model_df["deceased"])
    source_deceased_max_gradient_day = np.argmax(source_deceased_gradient)

    follow_up_deceased_gradient = np.gradient(follow_up_model_df["deceased"])
    follow_up_deceased_max_gradient_day = np.argmax(follow_up_deceased_gradient)

    test_a = (follow_up_deceased_max_gradient_day <
              source_deceased_max_gradient_day)

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_a, "MR4",
                         execution_time)

    assert test_a


def test_MR5():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(0, 0.999)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.mortality_time *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Check 1) (Total infected decreases by less than a factor of n)
    total_infected_delta = (source_model_df["total_infected"].iloc[-1] -
                            follow_up_model_df["total_infected"].iloc[-1])

    decreased_delta = (source_model_df["deceased"].iloc[-1] -
                       follow_up_model_df["deceased"].iloc[-1])

    # Does this cause a decrease in total infections?
    test_a = 0 < total_infected_delta

    # Does this cause a decrease in deceased?
    test_b = 0 < decreased_delta

    test_pass = test_a and test_b

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR5",
                         execution_time)
    assert test_a
    assert test_b


def test_MR6():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(1.001, 30)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.mortality_time *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Check 1) (Total infected decreases by less than a factor of n)
    total_infected_delta = (source_model_df["total_infected"].iloc[-1] -
                            follow_up_model_df["total_infected"].iloc[-1])

    deceased_delta = (source_model_df["deceased"].iloc[-1] -
                      follow_up_model_df["deceased"].iloc[-1])

    # Does this cause an increase in total infections?
    test_a = 0 > total_infected_delta

    # Does this cause an increase in deceased?
    test_b = 0 > deceased_delta

    test_pass = test_a and test_b
    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR6",
                         execution_time)
    assert test_a
    assert test_b


def test_MR7():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(0, 0.999)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.transmission_prob *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Check 1) (Total infected decreases by less than a factor of n)
    total_infected_delta = (source_model_df["total_infected"].iloc[-1] -
                            follow_up_model_df["total_infected"].iloc[-1])

    deceased_delta = (source_model_df["deceased"].iloc[-1] -
                      follow_up_model_df["deceased"].iloc[-1])

    recovered_delta = (source_model_df["recovered"].iloc[-1] -
                       follow_up_model_df["recovered"].iloc[-1])

    # Does this cause a decrease in total infections?
    test_a = 0 < total_infected_delta

    # Does this cause a decrease in deceased?
    test_b = 0 < deceased_delta

    # Does this cause a decrease in recovered?
    test_c = 0 < recovered_delta

    test_pass = test_a and test_b and test_c

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR7",
                         execution_time)
    assert test_a
    assert test_b
    assert test_c


def test_MR8():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(1.001, 30)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.transmission_prob *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Check 1) (Total infected decreases by less than a factor of n)
    total_infected_delta = (source_model_df["total_infected"].iloc[-1] -
                            follow_up_model_df["total_infected"].iloc[-1])

    deceased_delta = (source_model_df["deceased"].iloc[-1] -
                      follow_up_model_df["deceased"].iloc[-1])

    recovered_delta = (source_model_df["recovered"].iloc[-1] -
                       follow_up_model_df["recovered"].iloc[-1])

    # Does this cause an increase in total infections?
    test_a = 0 > total_infected_delta

    # Does this cause an increase in deceased?
    test_b = 0 > deceased_delta

    # Does this cause an increase in recovered?
    test_c = 0 > recovered_delta

    test_pass = test_a and test_b and test_c

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR8",
                         execution_time)
    assert test_a
    assert test_b
    assert test_c


def test_MR9():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(0, 0.999)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.incubation_time *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Crude check for pandemic duration: find day of maximum infection change
    d_source_total_infected_dt = np.gradient(source_model_df["total_infected"])
    d_source_total_infected_dt_max = np.argmax(d_source_total_infected_dt)

    d_follow_up_total_infected_dt = np.gradient(
        follow_up_model_df["total_infected"])
    d_follow_up_total_infected_dt_max = np.argmax(d_follow_up_total_infected_dt)

    total_infections_peak_rate_day_delta = (d_source_total_infected_dt_max -
                                            d_follow_up_total_infected_dt_max)

    # Should cause peak to occur sooner so day of source peak should occur later
    # than day of follow-up peak. Difference (source - follow-up) should
    # therefore be positive.
    test_a = (0 < total_infections_peak_rate_day_delta)

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_a, "MR9",
                         execution_time)
    assert test_a


def test_MR10():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(1.001, 30)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.incubation_time *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Crude check for pandemic duration: find day of maximum infection change
    d_source_total_infected_dt = np.gradient(source_model_df["total_infected"])
    d_source_total_infected_dt_max = np.argmax(d_source_total_infected_dt)

    d_follow_up_total_infected_dt = np.gradient(
        follow_up_model_df["total_infected"])
    d_follow_up_total_infected_dt_max = np.argmax(d_follow_up_total_infected_dt)

    total_infections_peak_rate_day_delta = (d_source_total_infected_dt_max -
                                            d_follow_up_total_infected_dt_max)

    # Should cause peak to occur later so day of source peak should occur sooner
    # than day of follow-up peak. Difference (source - follow-up) should
    # therefore be negative.
    test_a = (0 > total_infections_peak_rate_day_delta)

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_a, "MR10",
                         execution_time)
    assert test_a


def test_MR11():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(0, 0.999)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.recovery_time *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Get the difference in peak infections between source and follow-up
    peak_infectious_delta = (source_model_df["infectious"].max() -
                             follow_up_model_df["infectious"].max())

    # Get the gradient at each time step for both the source and follow-up
    source_deceased_gradient = np.gradient(source_model_df["deceased"])
    follow_up_decreased_gradient = np.gradient(follow_up_model_df["deceased"])

    # Get the difference in maximum gradient between source and follow-up
    deceased_gradient_delta = (max(source_deceased_gradient) -
                               max(follow_up_decreased_gradient))

    # Does this cause a decrease in peak infectious?
    test_a = 0 < peak_infectious_delta

    # Does this cause a decrease in rate of deceased?
    test_b = 0 < deceased_gradient_delta

    test_pass = test_a and test_b

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR11",
                         execution_time)
    assert test_a
    assert test_b


def test_MR12():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(1.001, 30)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.recovery_time *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Get the difference in peak infections between source and follow-up
    peak_infectious_delta = (source_model_df["infectious"].max() -
                             follow_up_model_df["infectious"].max())

    # Get the gradient at each time step for both the source and follow-up
    source_deceased_gradient = np.gradient(source_model_df["deceased"])
    follow_up_decreased_gradient = np.gradient(follow_up_model_df["deceased"])

    # Get the difference in maximum gradient between source and follow-up
    deceased_gradient_delta = (max(source_deceased_gradient) -
                               max(follow_up_decreased_gradient))

    # Does this cause an increase in peak infectious?
    test_a = 0 > peak_infectious_delta

    # Does this cause an increase in rate of deceased?
    test_b = 0 > deceased_gradient_delta

    test_pass = test_a and test_b

    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR12",
                         execution_time)
    assert test_a
    assert test_b


def test_MR13():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(0, 0.999)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.encounter_rate *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Get the difference in peak infections between source and follow-up
    peak_infectious_delta = (source_model_df["infectious"].max() -
                             follow_up_model_df["infectious"].max())

    # Get the gradient at each time step for both the source and follow-up
    source_deceased_gradient = np.gradient(source_model_df["deceased"])
    follow_up_decreased_gradient = np.gradient(follow_up_model_df["deceased"])

    # Get the difference in maximum gradient between source and follow-up
    deceased_gradient_delta = (max(source_deceased_gradient) -
                               max(follow_up_decreased_gradient))

    # Does this cause a decrease in peak infectious?
    test_a = 0 < peak_infectious_delta

    # Does this cause a decrease in rate of deceased?
    test_b = 0 < deceased_gradient_delta

    test_pass = test_a and test_b
    end_time = time.time()
    execution_time = end_time - start_time
    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR13",
                         execution_time)
    assert test_a
    assert test_b


def test_MR14():
    start_time = time.time()
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(1.001, 30)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.encounter_rate *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Get the difference in peak infections between source and follow-up
    peak_infectious_delta = (source_model_df["infectious"].max() -
                             follow_up_model_df["infectious"].max())

    # Get the gradient at each time step for both the source and follow-up
    source_deceased_gradient = np.gradient(source_model_df["deceased"])
    follow_up_decreased_gradient = np.gradient(follow_up_model_df["deceased"])

    # Get the difference in maximum gradient between source and follow-up
    deceased_gradient_delta = (max(source_deceased_gradient) -
                               max(follow_up_decreased_gradient))

    # Does this cause an increase in peak infectious?
    test_a = 0 > peak_infectious_delta

    # Does this cause an increase in rate of deceased?
    test_b = 0 > deceased_gradient_delta

    test_pass = test_a and test_b
    end_time = time.time()
    execution_time = end_time - start_time

    append_results_to_df(follow_up_model, follow_up_model_df, test_pass, "MR14",
                         execution_time)
    assert test_a
    assert test_b
