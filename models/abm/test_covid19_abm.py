"""
Metamorphic relations for the Spanish Influenza model as presented by Pullum
and Ozmen in their 2012 paper entitled: Early Results from Metamorphic Testing
of Epidemiological Models.

This test suite modifies and applies these metamorphic relations to Covasim,
a COVID-19 agent-based modelling tool.
"""

import numpy as np
import pandas as pd
import time
import pathlib
import os
import shutil

from covid19 import covid19, run_covid19
from itertools import product

OUTPUT_CSV_PATH = "models/abm/metamorphic_testing_results.csv"

np.random.seed(123)  # set seed for reproducibility

N_REPEATS = 30


def empty_results_df():
    inputs = ["mortality_prob", "recovery_time", "mortality_time",
              "transmission_prob", "encounter_rate", "incubation_time",
              "rand_seed"]
    outputs = ["cum_infections", "n_susceptible", "n_exposed",
               "n_infectious", "n_dead", "n_recovered"]
    simulation_days = list(range(0, 201))
    daily_outputs = [f"{o}_{n}" for (o, n) in product(outputs, simulation_days)]
    header = inputs + daily_outputs + ["input_delta_factor"]
    empty_df = pd.DataFrame(columns=header)
    return empty_df


def setup_module():
    # Create empty directories to store detailed MR results
    mrs = [f"mr{x}" for x in range(1, 15)]
    for mr in mrs:
        path_to_mr_dir = f"models/abm/mr_data/{mr}"
        if os.path.exists(path_to_mr_dir):
            shutil.rmtree(path_to_mr_dir)
        pathlib.Path(f"models/abm/mr_data/{mr}").mkdir(parents=True,
                                                       exist_ok=False)

    # Create an overall test results csv
    cols = ["relation", "input_changed", "output_affected",
            "input_delta_factor", "output_delta", "time", "repeats", "pass"]
    df = pd.DataFrame(columns=cols)
    df.to_csv(OUTPUT_CSV_PATH)


def append_execution_to_df(df, sim, relation, mt_type, ex_time,
                           input_delta_factor):
    results_dict = {}

    outputs = ["cum_infections", "n_susceptible",
               "n_exposed", "n_infectious",
               "n_dead", "n_recovered"]
    for output in outputs:
        for time_step, val in enumerate(sim.results[output]):
            full_results_col = f"{output}_{time_step}"
            results_dict[full_results_col] = val

    # Get value of each input from model
    results_dict["rand_seed"] = sim.pars["rand_seed"]
    results_dict["mortality_prob"] = sim.pars["prognoses"]["death_probs"]

    # Recovery time differs depending on symptom severity so collect all
    # recovery times.
    results_dict["recovery_time"] = {
        "asym2rec": sim.pars["dur"]["asym2rec"],
        "mild2rec": sim.pars["dur"]["mild2rec"],
        "sev2rec": sim.pars["dur"]["sev2rec"],
        "crit2rec": sim.pars["dur"]["crit2rec"]
    }

    # Death only occurs from critical disease state, hence only an individual
    # parameter.
    results_dict["mortality_time"] = sim.pars["dur"]["crit2die"]

    results_dict["transmission_prob"] = sim.pars["beta"]

    # Contacts per day varies based on contact layer, of which there are four,
    # so collect contacts per day for all contact layers.
    results_dict["encounter_rate"] = {
        "household": sim.pars["contacts"]["h"],
        "school": sim.pars["contacts"]["s"],
        "workplace": sim.pars["contacts"]["w"],
        "community": sim.pars["contacts"]["c"]
    }

    results_dict["pop_size"] = sim.pars["pop_size"]
    results_dict["pop_scale"] = sim.pars["pop_scale"]
    results_dict["pop_infected"] = sim.pars["pop_infected"]
    results_dict["incubation_time"] = sim.pars["dur"]["exp2inf"]
    results_dict["relation"] = relation
    results_dict["mt_type"] = mt_type
    results_dict["time"] = ex_time
    results_dict["input_delta_factor"] = input_delta_factor

    # Append inputs and outputs to existing df
    results_to_append_series = pd.Series(results_dict)
    df = df.append(results_to_append_series, ignore_index=True)
    return df


def append_results_to_df(df, relation, ex_time, test_pass, input, output,
                         input_delta, output_delta, repeats, source_input,
                         follow_up_input):
    results_dict = {"relation": relation,
                    "time": ex_time,
                    "pass": test_pass,
                    "input_changed": input,
                    "output_affected": output,
                    "input_delta_factor": input_delta,
                    "output_delta": output_delta,
                    "repeats": repeats,
                    "source_input": source_input,
                    "follow_up_input": follow_up_input}

    results_to_append_series = pd.Series(results_dict)
    df = df.append(results_to_append_series, ignore_index=True)
    return df


def run_source_test(df, mr):
    source_start_time = time.time()
    source_pars = {"rand_seed": np.random.randint(1, 1e6)}
    source_sim = covid19(pars_to_change=source_pars)
    executed_source_sim = run_covid19(source_sim)
    source_execution_time = time.time() - source_start_time
    execution_data_df = append_execution_to_df(df,
                                               executed_source_sim,
                                               mr,
                                               "source",
                                               source_execution_time,
                                               0)
    return execution_data_df


def save_execution_data_to_csv(df, mr):
    """Save execution data to the metamorphic relation directory.

    For repeat n of MR1, a file called models/abm/mr_data/mr1/repeat_n.csv is
    created. """
    _, _, files = next(os.walk(f"models/abm/mr_data/{mr}"))
    n_files = len(files)
    df.to_csv(f"models/abm/mr_data/{mr}/repeat_{n_files + 1}.csv")


def get_source_and_follow_up_df(df):
    """Return the source and follow-up executions from the execution df."""
    source_execution_data = df.loc[df["mt_type"] == "source"]
    follow_up_execution_data = df.loc[df["mt_type"] == "follow-up"]
    return source_execution_data, follow_up_execution_data


def test_MR1():
    """Decreasing mortality prob by a factor of n should decrease deaths by a
    factor of at most n."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(0, 0.999)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr1")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)
        follow_up_sim.pars["prognoses"]["death_probs"] *= n
        executed_follow_up_sim = run_covid19(follow_up_sim)
        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR1",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr1")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The mortality prob should be the same for all source executions
    source_mortality_prob = source_df["mortality_prob"].astype(str).unique()
    assert len(source_mortality_prob) == 1

    # The mortality prob should be the same for all follow-up executions
    follow_up_mortality_prob = follow_up_df[
        "mortality_prob"].astype(str).unique()
    assert len(follow_up_mortality_prob) == 1

    # The mortality prob should be different for source vs. follow-up executions
    assert (source_mortality_prob != follow_up_mortality_prob)
    ###########################################################################

    # Check 1) (Deceased decreases by less than a factor of n)
    deceased_delta = (source_df["n_dead_200"].mean() -
                      follow_up_df["n_dead_200"].mean())

    # Does this cause a decrease?
    test_a = 0 < deceased_delta

    # Check 2) (Is the magnitude of the effect a factor of at most n?)
    test_b = abs(follow_up_sim.pars["pop_size"] *
                 follow_up_sim.pars["pop_scale"] * n) > abs(deceased_delta)

    #
    recovered_delta = (source_df["n_recovered_200"].mean() -
                       follow_up_df["n_recovered_200"].mean())

    # Does this cause an increase?
    test_c = 0 > recovered_delta

    # Is the magnitude of the effect greater than 1/n?
    test_d = abs(follow_up_sim.pars["pop_size"] *
                 follow_up_sim.pars["pop_scale"] *
                 (1 / n)) < abs(recovered_delta)
    print(follow_up_sim.pars["pop_size"] *
                 follow_up_sim.pars["pop_scale"] *
                 (1 / n))
    print(abs(recovered_delta))

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b and test_c and test_d

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR1",
                                           execution_time,
                                           test_pass,
                                           "mortality_prob",
                                           "deceased",
                                           n,
                                           deceased_delta,
                                           n_repeats,
                                           source_mortality_prob,
                                           follow_up_mortality_prob)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_a and test_b and test_c and test_d


def test_MR1_fixed():
    """Decreasing mortality prob by a factor of n should decrease deaths by a
    factor of at most n."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(0, 0.999)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr1-fixed")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)
        follow_up_sim.pars["prognoses"]["death_probs"] *= n
        executed_follow_up_sim = run_covid19(follow_up_sim)
        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR1-fixed",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr1")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The mortality prob should be the same for all source executions
    source_mortality_prob = source_df["mortality_prob"].astype(str).unique()
    assert len(source_mortality_prob) == 1

    # The mortality prob should be the same for all follow-up executions
    follow_up_mortality_prob = follow_up_df[
        "mortality_prob"].astype(str).unique()
    assert len(follow_up_mortality_prob) == 1

    # The mortality prob should be different for source vs. follow-up executions
    assert (source_mortality_prob != follow_up_mortality_prob)
    ###########################################################################

    # Check 1) (Deceased decreases by less than a factor of n)
    deceased_delta = (source_df["n_dead_200"].mean() -
                      follow_up_df["n_dead_200"].mean())

    # Does this cause a decrease?
    test_a = 0 < deceased_delta

    # Check 2) (Is the magnitude of the effect a factor of at most n?)
    test_b = abs(follow_up_sim.pars["pop_size"] *
                 follow_up_sim.pars["pop_scale"] * n) > abs(deceased_delta)

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR1-fixed",
                                           execution_time,
                                           test_pass,
                                           "mortality_prob",
                                           "deceased",
                                           n,
                                           deceased_delta,
                                           n_repeats,
                                           source_mortality_prob,
                                           follow_up_mortality_prob)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_a and test_b


def test_MR2():
    """Increasing mortality prob by a factor of n should increase deaths by a
    factor of at most n."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(1.001, 30)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr2")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)
        follow_up_sim.pars["prognoses"]["death_probs"] *= n
        executed_follow_up_sim = run_covid19(follow_up_sim)
        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR2",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr2")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The mortality prob should be the same for all source executions
    source_mortality_prob = source_df["mortality_prob"].astype(str).unique()
    assert len(source_mortality_prob) == 1

    # The mortality prob should be the same for all follow-up executions
    follow_up_mortality_prob = follow_up_df[
        "mortality_prob"].astype(str).unique()
    assert len(follow_up_mortality_prob) == 1

    # The mortality prob should be different for source vs. follow-up executions
    assert (source_mortality_prob != follow_up_mortality_prob)
    ###########################################################################

    # Check 1) (Deceased increases by less than a factor of n)
    deceased_delta = (source_df["n_dead_200"].mean() -
                      follow_up_df["n_dead_200"].mean())

    # Does this cause an increase?
    test_a = 0 > deceased_delta

    # Check 2) (Is the magnitude of the effect a factor of at most n?)
    test_b = abs(follow_up_sim.pars["pop_size"] *
                 follow_up_sim.pars["pop_scale"] * n) > abs(deceased_delta)

    recovered_delta = (source_df["n_recovered_200"].mean() -
                       follow_up_df["n_recovered_200"].mean())

    # Does this cause a decrease?
    test_c = 0 < recovered_delta

    # Is the magnitude of the effect greater than 1/n?
    test_d = abs(follow_up_sim.pars["pop_size"] *
                 follow_up_sim.pars["pop_scale"] *
                 (1 / n)) < abs(recovered_delta)

    test_pass = test_a and test_b and test_c and test_d

    end_time = time.time()
    execution_time = end_time - start_time

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR2",
                                           execution_time,
                                           test_pass,
                                           "mortality_prob",
                                           "deceased",
                                           n,
                                           deceased_delta,
                                           n_repeats,
                                           source_mortality_prob,
                                           follow_up_mortality_prob)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR2_fixed():
    """Increasing mortality prob by a factor of n should increase deaths by a
    factor of at most n."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(1.001, 30)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr2-fixed")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)
        follow_up_sim.pars["prognoses"]["death_probs"] *= n
        executed_follow_up_sim = run_covid19(follow_up_sim)
        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR2-fixed",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr2")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The mortality prob should be the same for all source executions
    source_mortality_prob = source_df["mortality_prob"].astype(str).unique()
    assert len(source_mortality_prob) == 1

    # The mortality prob should be the same for all follow-up executions
    follow_up_mortality_prob = follow_up_df[
        "mortality_prob"].astype(str).unique()
    assert len(follow_up_mortality_prob) == 1

    # The mortality prob should be different for source vs. follow-up executions
    assert (source_mortality_prob != follow_up_mortality_prob)
    ###########################################################################

    # Check 1) (Deceased increases by less than a factor of n)
    deceased_delta = (source_df["n_dead_200"].mean() -
                      follow_up_df["n_dead_200"].mean())

    # Does this cause an increase?
    test_a = 0 > deceased_delta

    # Check 2) (Is the magnitude of the effect a factor of at most n?)
    test_b = abs(follow_up_sim.pars["pop_size"] *
                 follow_up_sim.pars["pop_scale"] * n) > abs(deceased_delta)

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR2-fixed",
                                           execution_time,
                                           test_pass,
                                           "mortality_prob",
                                           "deceased",
                                           n,
                                           deceased_delta,
                                           n_repeats,
                                           source_mortality_prob,
                                           follow_up_mortality_prob)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR3():
    """Decreasing the initial infectious population should delay the peak rise
    in deaths. We approximate this as the day at which the rate of change of
    deaths is at its greatest (i.e. index of the greatest gradient)."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(0, 0.999)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr3")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)
        follow_up_sim.pars["pop_infected"] *= n
        executed_follow_up_sim = run_covid19(follow_up_sim)
        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR3",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr3")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The pop infected should be the same for all source executions
    source_pop_infected = source_df["pop_infected"].unique()
    assert len(source_pop_infected) == 1

    # The pop infected should be the same for all follow-up executions
    follow_up_pop_infected = follow_up_df["pop_infected"].unique()
    assert len(follow_up_pop_infected) == 1

    # The pop infected should be different for source vs. follow-up executions
    assert (source_pop_infected != follow_up_pop_infected)
    ###########################################################################

    n_dead_cols = [f"n_dead_{day}" for day in range(0, 201)]
    n_dead_source_array = source_df[n_dead_cols].to_numpy()
    source_deceased_gradient = np.gradient(n_dead_source_array, axis=1)
    source_deceased_max_gradient_day = np.argmax(source_deceased_gradient,
                                                 axis=1)
    source_mean_max_gradient_day = source_deceased_max_gradient_day.mean()

    n_dead_follow_up_array = follow_up_df[n_dead_cols].to_numpy()
    follow_up_deceased_gradient = np.gradient(n_dead_follow_up_array, axis=1)
    follow_up_deceased_max_gradient_day = np.argmax(follow_up_deceased_gradient,
                                                    axis=1)
    follow_up_mean_max_gradient_day = follow_up_deceased_max_gradient_day.mean()

    peak_delta = (source_mean_max_gradient_day -
                  follow_up_mean_max_gradient_day)

    test_a = (follow_up_mean_max_gradient_day >
              source_mean_max_gradient_day)

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR3",
                                           execution_time,
                                           test_pass,
                                           "pop_infected",
                                           "peak_mortality_rate",
                                           n,
                                           peak_delta,
                                           n_repeats,
                                           source_pop_infected,
                                           follow_up_pop_infected)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR4():
    """Increasing the initially infectious population should make the peak rise
    in deaths occur earlier. We approximate this as the day at which the rate of
    change of deaths is at its greatest (i.e. index of the greatest gradient).
    """
    # Repeat every test 30 times using the same change in input parameter.

    # The change for this test case is capped at 10x. Due to the scaling factor,
    # if the population size is increased by much more than 10x, Covasim throws
    # an error as it infects more individuals than there are individuals in the
    # population.
    n = np.random.uniform(1.001, 10)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr4")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        follow_up_sim.pars["pop_infected"] *= n
        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR4",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr4")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The pop infected should be the same for all source executions
    source_pop_infected = source_df["pop_infected"].unique()
    assert len(source_pop_infected) == 1

    # The pop infected should be the same for all follow-up executions
    follow_up_pop_infected = follow_up_df["pop_infected"].unique()
    assert len(follow_up_pop_infected) == 1

    # The pop infected should be different for source vs. follow-up executions
    assert (source_pop_infected != follow_up_pop_infected)
    ###########################################################################

    n_dead_cols = [f"n_dead_{day}" for day in range(0, 201)]
    n_dead_source_array = source_df[n_dead_cols].to_numpy()
    source_deceased_gradient = np.gradient(n_dead_source_array, axis=1)
    source_deceased_max_gradient_day = np.argmax(source_deceased_gradient,
                                                 axis=1)
    source_mean_max_gradient_day = source_deceased_max_gradient_day.mean()

    n_dead_follow_up_array = follow_up_df[n_dead_cols].to_numpy()
    follow_up_deceased_gradient = np.gradient(n_dead_follow_up_array, axis=1)
    follow_up_deceased_max_gradient_day = np.argmax(follow_up_deceased_gradient,
                                                    axis=1)
    follow_up_mean_max_gradient_day = follow_up_deceased_max_gradient_day.mean()

    peak_delta = (source_mean_max_gradient_day -
                  follow_up_mean_max_gradient_day)

    test_a = (follow_up_mean_max_gradient_day <
              source_mean_max_gradient_day)

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR4",
                                           execution_time,
                                           test_pass,
                                           "pop_infected",
                                           "peak_mortality_rate",
                                           n,
                                           peak_delta,
                                           n_repeats,
                                           source_pop_infected,
                                           follow_up_pop_infected)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR5():
    """Decreasing the death time should reduce the number of infections and
    deaths."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(0, 0.999)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr5")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Decrease mean of the mortality time
        follow_up_sim.pars["dur"]["crit2die"]["par1"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR5",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr5")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The mortality time should be the same for all source executions
    source_mortality_time = source_df["mortality_time"].astype(str).unique()
    assert len(source_mortality_time) == 1

    # The mortality time should be the same for all follow-up executions
    follow_up_mortality_time = follow_up_df[
        "mortality_time"].astype(str).unique()
    assert len(follow_up_mortality_time) == 1

    # The mortality time should be different for source vs. follow-up executions
    assert (source_mortality_time != follow_up_mortality_time)
    ###########################################################################

    infected_delta = (source_df["cum_infections_200"].mean() -
                      follow_up_df["cum_infections_200"].mean())
    deceased_delta = (source_df["n_dead_200"].mean() -
                      follow_up_df["n_dead_200"].mean())

    test_a = 0 < infected_delta
    test_b = 0 < deceased_delta

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR5",
                                           execution_time,
                                           test_pass,
                                           "mortality_time_mean",
                                           ["deceased", "infections"],
                                           n,
                                           {"infected_delta": infected_delta,
                                            "deceased_delta": deceased_delta},
                                           n_repeats,
                                           source_mortality_time,
                                           follow_up_mortality_time)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


# def test_MR5_fixed():
#     """Decreasing the death time should reduce the number of infections and
#     deaths."""
#     # Repeat every test 30 times using the same change in input parameter.
#     n = np.random.uniform(0, 0.999)
#     execution_data_df = empty_results_df()
#     results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])
#
#     # Time completion of n repeats
#     start_time = time.time()
#     n_repeats = N_REPEATS
#     for repeat in range(n_repeats):
#         execution_data_df = run_source_test(execution_data_df, "mr5")
#
#         follow_up_start_time = time.time()
#         follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
#         follow_up_sim = covid19(pars_to_change=follow_up_pars)
#
#         # Decrease mean of the mortality time AND recovery times
#         follow_up_sim.pars["dur"]["crit2die"]["par1"] *= n
#         follow_up_sim.pars["dur"]["asym2rec"]["par1"] *= n
#         follow_up_sim.pars["dur"]["mild2rec"]["par1"] *= n
#         follow_up_sim.pars["dur"]["sev2rec"]["par1"] *= n
#         follow_up_sim.pars["dur"]["crit2rec"]["par1"] *= n
#
#         executed_follow_up_sim = run_covid19(follow_up_sim)
#
#         follow_up_execution_time = time.time() - follow_up_start_time
#         execution_data_df = append_execution_to_df(execution_data_df,
#                                                    executed_follow_up_sim,
#                                                    "MR5",
#                                                    "follow-up",
#                                                    follow_up_execution_time,
#                                                    n)
#
#     save_execution_data_to_csv(execution_data_df, "mr5")
#     source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)
#
#     ############################## SANITY CHECKS ##############################
#     # The mortality time should be the same for all source executions
#     source_mortality_time = source_df["mortality_time"].astype(str).unique()
#     assert len(source_mortality_time) == 1
#
#     # The mortality time should be the same for all follow-up executions
#     follow_up_mortality_time = follow_up_df[
#         "mortality_time"].astype(str).unique()
#     assert len(follow_up_mortality_time) == 1
#
#     # The mortality time should be different for source vs. follow-up executions
#     assert (source_mortality_time != follow_up_mortality_time)
#     ###########################################################################
#
#     infected_delta = (source_df["cum_infections_200"].mean() -
#                       follow_up_df["cum_infections_200"].mean())
#     deceased_delta = (source_df["n_dead_200"].mean() -
#                       follow_up_df["n_dead_200"].mean())
#
#     test_a = 0 < infected_delta
#     test_b = 0 < deceased_delta
#
#     end_time = time.time()
#     execution_time = end_time - start_time
#     test_pass = test_a and test_b
#
#     # Write the overall metamorphic test results to CSV
#     results_data_df = append_results_to_df(results_data_df,
#                                            "MR5",
#                                            execution_time,
#                                            test_pass,
#                                            "mortality_time_mean",
#                                            ["deceased", "infections"],
#                                            n,
#                                            {"infected_delta": infected_delta,
#                                             "deceased_delta": deceased_delta},
#                                            n_repeats,
#                                            source_mortality_time,
#                                            follow_up_mortality_time)
#
#     results_data_df.to_csv(OUTPUT_CSV_PATH)
#     assert test_pass


def test_MR6():
    """Increasing the death time should increase the number of infections and
    deaths."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(1.001, 30)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr6")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Increase mean of the mortality time
        follow_up_sim.pars["dur"]["crit2die"]["par1"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR6",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr6")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The mortality time should be the same for all source executions
    source_mortality_time = source_df["mortality_time"].astype(str).unique()
    assert len(source_mortality_time) == 1

    # The mortality time should be the same for all follow-up executions
    follow_up_mortality_time = follow_up_df[
        "mortality_time"].astype(str).unique()
    assert len(follow_up_mortality_time) == 1

    # The mortality time should be different for source vs. follow-up executions
    assert (source_mortality_time != follow_up_mortality_time)
    ###########################################################################

    infected_delta = (source_df["cum_infections_200"].mean() -
                      follow_up_df["cum_infections_200"].mean())
    deceased_delta = (source_df["n_dead_200"].mean() -
                      follow_up_df["n_dead_200"].mean())

    test_a = 0 > infected_delta
    test_b = 0 > deceased_delta

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR6",
                                           execution_time,
                                           test_pass,
                                           "mortality_time_mean",
                                           ["deceased", "infections"],
                                           n,
                                           {"infected_delta": infected_delta,
                                            "deceased_delta": deceased_delta},
                                           n_repeats,
                                           source_mortality_time,
                                           follow_up_mortality_time)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


# def test_MR6_fixed():
#     """Increasing the death time should increase the number of infections and
#     deaths."""
#     # Repeat every test 30 times using the same change in input parameter.
#     n = np.random.uniform(1.001, 30)
#     execution_data_df = empty_results_df()
#     results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])
#
#     # Time completion of n repeats
#     start_time = time.time()
#     n_repeats = N_REPEATS
#     for repeat in range(n_repeats):
#         execution_data_df = run_source_test(execution_data_df, "mr6")
#
#         follow_up_start_time = time.time()
#         follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
#         follow_up_sim = covid19(pars_to_change=follow_up_pars)
#
#         # Increase mean of the mortality AND recovery times
#         follow_up_sim.pars["dur"]["crit2die"]["par1"] *= n
#         follow_up_sim.pars["dur"]["asym2rec"]["par1"] *= n
#         follow_up_sim.pars["dur"]["mild2rec"]["par1"] *= n
#         follow_up_sim.pars["dur"]["sev2rec"]["par1"] *= n
#         follow_up_sim.pars["dur"]["crit2rec"]["par1"] *= n
#
#         executed_follow_up_sim = run_covid19(follow_up_sim)
#
#         follow_up_execution_time = time.time() - follow_up_start_time
#         execution_data_df = append_execution_to_df(execution_data_df,
#                                                    executed_follow_up_sim,
#                                                    "MR6",
#                                                    "follow-up",
#                                                    follow_up_execution_time,
#                                                    n)
#
#     save_execution_data_to_csv(execution_data_df, "mr6")
#     source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)
#
#     ############################## SANITY CHECKS ##############################
#     # The mortality time should be the same for all source executions
#     source_mortality_time = source_df["mortality_time"].astype(str).unique()
#     assert len(source_mortality_time) == 1
#
#     # The mortality time should be the same for all follow-up executions
#     follow_up_mortality_time = follow_up_df[
#         "mortality_time"].astype(str).unique()
#     assert len(follow_up_mortality_time) == 1
#
#     # The mortality time should be different for source vs. follow-up executions
#     assert (source_mortality_time != follow_up_mortality_time)
#     ###########################################################################
#
#     infected_delta = (source_df["cum_infections_200"].mean() -
#                       follow_up_df["cum_infections_200"].mean())
#     deceased_delta = (source_df["n_dead_200"].mean() -
#                       follow_up_df["n_dead_200"].mean())
#
#     test_a = 0 > infected_delta
#     test_b = 0 > deceased_delta
#
#     end_time = time.time()
#     execution_time = end_time - start_time
#     test_pass = test_a and test_b
#
#     # Write the overall metamorphic test results to CSV
#     results_data_df = append_results_to_df(results_data_df,
#                                            "MR6",
#                                            execution_time,
#                                            test_pass,
#                                            "mortality_time_mean",
#                                            ["deceased", "infections"],
#                                            n,
#                                            {"infected_delta": infected_delta,
#                                             "deceased_delta": deceased_delta},
#                                            n_repeats,
#                                            source_mortality_time,
#                                            follow_up_mortality_time)
#
#     results_data_df.to_csv(OUTPUT_CSV_PATH)
#     assert test_pass


def test_MR7():
    """Decreasing transmission prob (beta) should decrease infections,
    deaths, and recoveries."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(0, 0.999)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr7")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Decrease transmission probability
        follow_up_sim.pars["beta"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR7",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr7")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The transmission probability should be the same for all source executions
    source_beta = source_df["transmission_prob"].unique()
    assert len(source_beta) == 1

    # The transmission probability should be the same for all follow-up
    # executions
    follow_up_beta = follow_up_df["transmission_prob"].unique()
    assert len(follow_up_beta) == 1

    # The transmission probability should be different for source vs. follow-up
    # executions
    assert (source_beta != follow_up_beta)
    ###########################################################################

    infected_delta = (source_df["cum_infections_200"].mean() -
                      follow_up_df["cum_infections_200"].mean())
    deceased_delta = (source_df["n_dead_200"].mean() -
                      follow_up_df["n_dead_200"].mean())
    recovered_delta = (source_df["n_recovered_200"].mean() -
                       follow_up_df["n_recovered_200"].mean())

    test_a = 0 < infected_delta
    test_b = 0 < deceased_delta
    test_c = 0 < recovered_delta

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b and test_c

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR7",
                                           execution_time,
                                           test_pass,
                                           "transmission_prob",
                                           ["infected",
                                            "deceased",
                                            "recovered"],
                                           n,
                                           {"infected_delta": infected_delta,
                                            "deceased_delta": deceased_delta,
                                            "recovered_delta": recovered_delta},
                                           n_repeats,
                                           source_beta,
                                           follow_up_beta)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR8():
    """Increasing transmission prob (beta) should increase infections,
    deaths, and recoveries."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(1.001, 30)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr8")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Increase transmission probability
        follow_up_sim.pars["beta"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR8",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr8")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The transmission probability should be the same for all source executions
    source_beta = source_df["transmission_prob"].unique()
    assert len(source_beta) == 1

    # The transmission probability should be the same for all follow-up
    # executions
    follow_up_beta = follow_up_df["transmission_prob"].unique()
    assert len(follow_up_beta) == 1

    # The transmission probability should be different for source vs. follow-up
    # executions
    assert (source_beta != follow_up_beta)
    ###########################################################################

    infected_delta = (source_df["cum_infections_200"].mean() -
                      follow_up_df["cum_infections_200"].mean())
    deceased_delta = (source_df["n_dead_200"].mean() -
                      follow_up_df["n_dead_200"].mean())
    recovered_delta = (source_df["n_recovered_200"].mean() -
                       follow_up_df["n_recovered_200"].mean())

    test_a = 0 > infected_delta
    test_b = 0 > deceased_delta
    test_c = 0 > recovered_delta

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b and test_c

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR8",
                                           execution_time,
                                           test_pass,
                                           "transmission_prob",
                                           ["infected",
                                            "deceased",
                                            "recovered"],
                                           n,
                                           {"infected_delta": infected_delta,
                                            "deceased_delta": deceased_delta,
                                            "recovered_delta": recovered_delta},
                                           n_repeats,
                                           source_beta,
                                           follow_up_beta)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR9():
    """Decreasing incubation time should cause peak infections to occur sooner.
    This means that the day of the source peak should occur later than day of
    follow-up peak. The difference (source - follow-up) should therefore be
    positive."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(0, 0.999)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr9")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Decrease incubation time
        follow_up_sim.pars["dur"]["exp2inf"]["par1"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR9",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr9")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The incubation time should be the same for all source executions
    source_incubation_time = source_df["incubation_time"].astype(str).unique()
    assert len(source_incubation_time) == 1

    # The incubation time should be the same for all follow-up executions
    follow_up_incubation_time = follow_up_df[
        "incubation_time"].astype(str).unique()
    assert len(follow_up_incubation_time) == 1

    # The incubation time should be different for source vs. follow-up executions
    assert (source_incubation_time != follow_up_incubation_time)
    ###########################################################################

    cum_infections_cols = [f"cum_infections_{day}" for day in range(0, 201)]
    cum_infections_source_array = source_df[
        cum_infections_cols].to_numpy()
    source_cum_infections_gradients = np.gradient(cum_infections_source_array,
                                                  axis=1)

    source_cum_infections_max_gradient_days = np.argmax(
        source_cum_infections_gradients, axis=1)

    mean_source_cum_infections_max_gradient_day = \
        source_cum_infections_max_gradient_days.mean()

    cum_infections_follow_up_array = follow_up_df[
        cum_infections_cols].to_numpy()
    follow_up_cum_infections_gradients = np.gradient(
        cum_infections_follow_up_array, axis=1)
    follow_up_cum_infections_max_gradient_days = np.argmax(
        follow_up_cum_infections_gradients, axis=1)

    mean_follow_up_cum_infections_max_gradient_day = \
        follow_up_cum_infections_max_gradient_days.mean()

    peak_day_delta = (mean_source_cum_infections_max_gradient_day -
                      mean_follow_up_cum_infections_max_gradient_day)

    test_a = peak_day_delta > 0

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR9",
                                           execution_time,
                                           test_pass,
                                           "incubation_time",
                                           "peak_cum_infections_rate",
                                           n,
                                           peak_day_delta,
                                           n_repeats,
                                           source_incubation_time,
                                           follow_up_incubation_time)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR10():
    """Increasing incubation time should delay peak infection rate."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(1.001, 30)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr10")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Increase incubation time
        follow_up_sim.pars["dur"]["exp2inf"]["par1"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR10",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr10")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The incubation time should be the same for all source executions
    source_incubation_time = source_df["incubation_time"].astype(str).unique()
    assert len(source_incubation_time) == 1

    # The incubation time should be the same for all follow-up executions
    follow_up_incubation_time = follow_up_df[
        "incubation_time"].astype(str).unique()
    assert len(follow_up_incubation_time) == 1

    # The incubation time should be different for source vs. follow-up executions
    assert (source_incubation_time != follow_up_incubation_time)
    ###########################################################################

    cum_infections_cols = [f"cum_infections_{day}" for day in range(0, 201)]
    cum_infections_source_array = source_df[
        cum_infections_cols].to_numpy()
    source_cum_infections_gradients = np.gradient(cum_infections_source_array,
                                                  axis=1)

    source_cum_infections_max_gradient_days = np.argmax(
        source_cum_infections_gradients, axis=1)

    mean_source_cum_infections_max_gradient_day = \
        source_cum_infections_max_gradient_days.mean()

    cum_infections_follow_up_array = follow_up_df[
        cum_infections_cols].to_numpy()
    follow_up_cum_infections_gradients = np.gradient(
        cum_infections_follow_up_array, axis=1)
    follow_up_cum_infections_max_gradient_days = np.argmax(
        follow_up_cum_infections_gradients, axis=1)

    mean_follow_up_cum_infections_max_gradient_day = \
        follow_up_cum_infections_max_gradient_days.mean()

    peak_day_delta = (mean_source_cum_infections_max_gradient_day -
                      mean_follow_up_cum_infections_max_gradient_day)

    test_a = peak_day_delta < 0

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR10",
                                           execution_time,
                                           test_pass,
                                           "incubation_time",
                                           "peak_cum_infections_rate_delta",
                                           n,
                                           peak_day_delta,
                                           n_repeats,
                                           source_incubation_time,
                                           follow_up_incubation_time)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR11():
    """Decreasing recovery time should decrease peak infections and, in turn,
    reduce the maximum rate of death."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(0, 0.999)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr11")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Decrease recovery times
        follow_up_sim.pars["dur"]["asym2rec"]["par1"] *= n
        follow_up_sim.pars["dur"]["mild2rec"]["par1"] *= n
        follow_up_sim.pars["dur"]["sev2rec"]["par1"] *= n
        follow_up_sim.pars["dur"]["crit2rec"]["par1"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR11",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr11")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The recovery time should be the same for all source executions
    source_recovery_time = source_df["recovery_time"].astype(str).unique()
    assert len(source_recovery_time) == 1

    # The recovery time should be the same for all follow-up executions
    follow_up_recovery_time = follow_up_df[
        "recovery_time"].astype(str).unique()
    assert len(follow_up_recovery_time) == 1

    # The recovery time should be different for source vs. follow-up executions
    assert (source_recovery_time != follow_up_recovery_time)
    ###########################################################################

    cum_infections_cols = [f"cum_infections_{day}" for day in range(0, 201)]

    # Get cumulative infections time-series for source and follow-up
    cum_infections_source_array = source_df[
        cum_infections_cols].to_numpy()
    cum_infections_follow_up_array = follow_up_df[
        cum_infections_cols].to_numpy()

    source_peak_infections = cum_infections_source_array.max(axis=1)
    source_mean_peak_infections = source_peak_infections.mean()
    follow_up_peak_infections = cum_infections_follow_up_array.max(axis=1)
    follow_up_mean_peak_infections = follow_up_peak_infections.mean()

    # Get difference in peak infections between source and follow-up
    peak_infections_delta = (source_mean_peak_infections -
                             follow_up_mean_peak_infections)

    test_a = peak_infections_delta > 0

    # Get difference in maximum rate of death between source and follow-up
    n_dead_cols = [f"n_dead_{day}" for day in range(0, 201)]
    n_dead_source_array = source_df[n_dead_cols].to_numpy()
    source_deceased_gradient = np.gradient(n_dead_source_array, axis=1)
    source_deceased_max_gradient = np.max(source_deceased_gradient, axis=1)
    mean_source_max_gradient = source_deceased_max_gradient.mean()

    n_dead_follow_up_array = follow_up_df[n_dead_cols].to_numpy()
    follow_up_deceased_gradient = np.gradient(n_dead_follow_up_array, axis=1)
    follow_up_deceased_max_gradient = np.max(follow_up_deceased_gradient,
                                             axis=1)
    mean_follow_up_max_gradient = follow_up_deceased_max_gradient.mean()

    peak_death_rate_delta = (mean_source_max_gradient -
                             mean_follow_up_max_gradient)

    test_b = peak_death_rate_delta > 0

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR11",
                                           execution_time,
                                           test_pass,
                                           "recovery_time",
                                           ["peak_cum_infections_delta",
                                            "peak_death_rate_delta"],
                                           n,
                                           {"peak_cum_infections_delta":
                                                peak_infections_delta,
                                            "peak_death_rate_delta":
                                                peak_death_rate_delta},
                                           n_repeats,
                                           source_recovery_time,
                                           follow_up_recovery_time)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR12():
    """Increasing incubation time should increase peak infections and, in turn,
    increase the maximum rate of death."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(1.001, 30)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr12")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Increase recovery times
        follow_up_sim.pars["dur"]["asym2rec"]["par1"] *= n
        follow_up_sim.pars["dur"]["mild2rec"]["par1"] *= n
        follow_up_sim.pars["dur"]["sev2rec"]["par1"] *= n
        follow_up_sim.pars["dur"]["crit2rec"]["par1"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR12",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr12")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The recovery time should be the same for all source executions
    source_recovery_time = source_df["recovery_time"].astype(str).unique()
    assert len(source_recovery_time) == 1

    # The recovery time should be the same for all follow-up executions
    follow_up_recovery_time = follow_up_df[
        "recovery_time"].astype(str).unique()
    assert len(follow_up_recovery_time) == 1

    # The recovery time should be different for source vs. follow-up executions
    assert (source_recovery_time != follow_up_recovery_time)
    ###########################################################################

    cum_infections_cols = [f"cum_infections_{day}" for day in range(0, 201)]

    # Get cumulative infections time-series for source and follow-up
    cum_infections_source_array = source_df[
        cum_infections_cols].to_numpy()
    cum_infections_follow_up_array = follow_up_df[
        cum_infections_cols].to_numpy()

    source_peak_infections = cum_infections_source_array.max(axis=1)
    source_mean_peak_infections = source_peak_infections.mean()
    follow_up_peak_infections = cum_infections_follow_up_array.max(axis=1)
    follow_up_mean_peak_infections = follow_up_peak_infections.mean()

    # Get difference in peak infections between source and follow-up
    peak_infections_delta = (source_mean_peak_infections -
                             follow_up_mean_peak_infections)

    test_a = peak_infections_delta < 0

    # Get difference in maximum rate of death between source and follow-up
    n_dead_cols = [f"n_dead_{day}" for day in range(0, 201)]
    n_dead_source_array = source_df[n_dead_cols].to_numpy()
    source_deceased_gradient = np.gradient(n_dead_source_array, axis=1)
    source_deceased_max_gradient = np.max(source_deceased_gradient, axis=1)
    mean_source_max_gradient = source_deceased_max_gradient.mean()

    n_dead_follow_up_array = follow_up_df[n_dead_cols].to_numpy()
    follow_up_deceased_gradient = np.gradient(n_dead_follow_up_array, axis=1)
    follow_up_deceased_max_gradient = np.max(follow_up_deceased_gradient,
                                             axis=1)
    mean_follow_up_max_gradient = follow_up_deceased_max_gradient.mean()

    peak_death_rate_delta = (mean_source_max_gradient -
                             mean_follow_up_max_gradient)

    test_b = peak_death_rate_delta < 0

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR12",
                                           execution_time,
                                           test_pass,
                                           "recovery_time",
                                           ["peak_cum_infections_delta",
                                            "peak_death_rate_delta"],
                                           n,
                                           {"peak_cum_infections_delta":
                                                peak_infections_delta,
                                            "peak_death_rate_delta":
                                                peak_death_rate_delta},
                                           n_repeats,
                                           source_recovery_time,
                                           follow_up_recovery_time)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR13():
    """Decreasing encounter rate should decrease peak infections and, in turn,
    decrease the maximum rate of death."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(0, 0.999)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr13")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Decrease contacts
        follow_up_sim.pars["contacts"]["h"] *= n
        follow_up_sim.pars["contacts"]["s"] *= n
        follow_up_sim.pars["contacts"]["w"] *= n
        follow_up_sim.pars["contacts"]["c"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR13",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr13")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The encounter rate should be the same for all source executions
    source_encounter_rate = source_df["encounter_rate"].astype(str).unique()
    assert len(source_encounter_rate) == 1

    # The encounter rate should be the same for all follow-up executions
    follow_up_encounter_rate = follow_up_df[
        "encounter_rate"].astype(str).unique()
    assert len(follow_up_encounter_rate) == 1

    # The encounter rate should be different for source vs. follow-up executions
    assert (source_encounter_rate != follow_up_encounter_rate)
    ###########################################################################

    cum_infections_cols = [f"cum_infections_{day}" for day in range(0, 201)]

    # Get cumulative infections time-series for source and follow-up
    cum_infections_source_array = source_df[
        cum_infections_cols].to_numpy()
    cum_infections_follow_up_array = follow_up_df[
        cum_infections_cols].to_numpy()

    # Get difference in average peak infections between source and follow-up
    peak_cum_infections_source = cum_infections_source_array.max(axis=1)
    mean_peak_cum_infections_source = peak_cum_infections_source.mean()

    peak_cum_infections_follow_up = cum_infections_follow_up_array.max(axis=1)
    mean_peak_cum_infections_follow_up = peak_cum_infections_follow_up.mean()

    peak_infections_delta = (mean_peak_cum_infections_source -
                             mean_peak_cum_infections_follow_up)

    test_a = peak_infections_delta > 0

    # Get difference in maximum rate of death between source and follow-up
    n_dead_cols = [f"n_dead_{day}" for day in range(0, 201)]
    n_dead_source_array = source_df[n_dead_cols].to_numpy()
    source_deceased_gradient = np.gradient(n_dead_source_array, axis=1)
    source_deceased_max_gradient = np.max(source_deceased_gradient, axis=1)
    mean_source_max_gradient = source_deceased_max_gradient.mean()

    n_dead_follow_up_array = follow_up_df[n_dead_cols].to_numpy()
    follow_up_deceased_gradient = np.gradient(n_dead_follow_up_array, axis=1)
    follow_up_deceased_max_gradient = np.max(follow_up_deceased_gradient,
                                             axis=1)
    mean_follow_up_max_gradient = follow_up_deceased_max_gradient.mean()

    peak_death_rate_delta = (mean_source_max_gradient -
                             mean_follow_up_max_gradient)

    test_b = peak_death_rate_delta > 0

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR13",
                                           execution_time,
                                           test_pass,
                                           "encounter_rate",
                                           ["peak_cum_infections_delta",
                                            "peak_death_rate_delta"],
                                           n,
                                           {"peak_cum_infections_delta":
                                                peak_infections_delta,
                                            "peak_death_rate_delta":
                                                peak_death_rate_delta},
                                           n_repeats,
                                           source_encounter_rate,
                                           follow_up_encounter_rate)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass


def test_MR14():
    """Increasing encounter rate should increase peak infections and, in turn,
    increase the maximum rate of death."""
    # Repeat every test 30 times using the same change in input parameter.
    n = np.random.uniform(1.001, 30)
    execution_data_df = empty_results_df()
    results_data_df = pd.read_csv(OUTPUT_CSV_PATH, index_col=[0])

    # Time completion of n repeats
    start_time = time.time()
    n_repeats = N_REPEATS
    for repeat in range(n_repeats):
        execution_data_df = run_source_test(execution_data_df, "mr14")

        follow_up_start_time = time.time()
        follow_up_pars = {"rand_seed": np.random.randint(1, 1e6)}
        follow_up_sim = covid19(pars_to_change=follow_up_pars)

        # Decrease contacts
        follow_up_sim.pars["contacts"]["h"] *= n
        follow_up_sim.pars["contacts"]["s"] *= n
        follow_up_sim.pars["contacts"]["w"] *= n
        follow_up_sim.pars["contacts"]["c"] *= n

        executed_follow_up_sim = run_covid19(follow_up_sim)

        follow_up_execution_time = time.time() - follow_up_start_time
        execution_data_df = append_execution_to_df(execution_data_df,
                                                   executed_follow_up_sim,
                                                   "MR14",
                                                   "follow-up",
                                                   follow_up_execution_time,
                                                   n)

    save_execution_data_to_csv(execution_data_df, "mr14")
    source_df, follow_up_df = get_source_and_follow_up_df(execution_data_df)

    ############################## SANITY CHECKS ##############################
    # The encounter rate should be the same for all source executions
    source_encounter_rate = source_df["encounter_rate"].astype(str).unique()
    assert len(source_encounter_rate) == 1

    # The encounter rate should be the same for all follow-up executions
    follow_up_encounter_rate = follow_up_df[
        "encounter_rate"].astype(str).unique()
    assert len(follow_up_encounter_rate) == 1

    # The encounter rate should be different for source vs. follow-up executions
    assert (source_encounter_rate != follow_up_encounter_rate)
    ###########################################################################

    cum_infections_cols = [f"cum_infections_{day}" for day in range(0, 201)]

    # Get cumulative infections time-series for source and follow-up
    cum_infections_source_array = source_df[
        cum_infections_cols].to_numpy()
    cum_infections_follow_up_array = follow_up_df[
        cum_infections_cols].to_numpy()

    # Get difference in average peak infections between source and follow-up
    peak_cum_infections_source = cum_infections_source_array.max(axis=1)
    mean_peak_cum_infections_source = peak_cum_infections_source.mean()

    peak_cum_infections_follow_up = cum_infections_follow_up_array.max(axis=1)
    mean_peak_cum_infections_follow_up = peak_cum_infections_follow_up.mean()

    peak_infections_delta = (mean_peak_cum_infections_source -
                             mean_peak_cum_infections_follow_up)

    test_a = peak_infections_delta < 0

    # Get difference in maximum rate of death between source and follow-up
    n_dead_cols = [f"n_dead_{day}" for day in range(0, 201)]
    n_dead_source_array = source_df[n_dead_cols].to_numpy()
    source_deceased_gradient = np.gradient(n_dead_source_array, axis=1)
    source_deceased_max_gradient = np.max(source_deceased_gradient, axis=1)
    mean_source_max_gradient = source_deceased_max_gradient.mean()

    n_dead_follow_up_array = follow_up_df[n_dead_cols].to_numpy()
    follow_up_deceased_gradient = np.gradient(n_dead_follow_up_array, axis=1)
    follow_up_deceased_max_gradient = np.max(follow_up_deceased_gradient,
                                             axis=1)
    mean_follow_up_max_gradient = follow_up_deceased_max_gradient.mean()

    peak_death_rate_delta = (mean_source_max_gradient -
                             mean_follow_up_max_gradient)

    test_b = peak_death_rate_delta < 0

    end_time = time.time()
    execution_time = end_time - start_time
    test_pass = test_a and test_b

    # Write the overall metamorphic test results to CSV
    results_data_df = append_results_to_df(results_data_df,
                                           "MR14",
                                           execution_time,
                                           test_pass,
                                           "encounter_rate",
                                           ["peak_cum_infections_delta",
                                            "peak_death_rate_delta"],
                                           n,
                                           {"peak_cum_infections_delta":
                                                peak_infections_delta,
                                            "peak_death_rate_delta":
                                                peak_death_rate_delta},
                                           n_repeats,
                                           source_encounter_rate,
                                           follow_up_encounter_rate)

    results_data_df.to_csv(OUTPUT_CSV_PATH)
    assert test_pass
