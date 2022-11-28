"""
Metamorphic relations for the Spanish Influenza model as presented by Pullum
and Ozmen in their 2012 paper entitled: Early Results from Metamorphic Testing
of Epidemiological Models.
"""

import numpy as np

from spanish_influenza_ebm import SpanishInfluenzaEBM


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


def test_MR1():
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=True)

    n = np.random.uniform(0, 0.999)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.mortality_prob *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=True)

    # Check 1) (Deceased decreases by less than a factor of n)
    deceased_delta = (source_model_df["deceased"].iloc[-1] -
                      follow_up_model_df["deceased"].iloc[-1])

    # Does this cause a decrease?
    assert 0 < deceased_delta

    # Is the magnitude of the effect a factor of at most n?
    assert abs(source_model.population_size*n) > abs(deceased_delta)

    # Check 2) (Recovered increases by more than 1/n)
    # NOTE: error in Table 1 of the original paper; row 2 of MR1 should say
    # increases the number of recovered by more than 1/n, not decrease. It
    # doesn't make sense for a reduction in mortality probability to cause a
    # decrease in recoveries.
    recovered_delta = (source_model_df["recovered"].iloc[-1] -
                       follow_up_model_df["recovered"].iloc[-1])

    # Does this cause an increase?
    assert 0 > recovered_delta

    # Is the magnitude of the effect greater than 1/n?
    assert abs(1/n) < abs(recovered_delta)


def test_MR2():
    source_model = SpanishInfluenzaEBM()
    source_model_df = source_model.solve(plot=True)

    n = np.random.uniform(1.001, 10)
    follow_up_model = SpanishInfluenzaEBM()
    follow_up_model.mortality_prob *= n  # Apply the intervention
    follow_up_model_df = follow_up_model.solve(plot=True)

    # Check 1) (Deceased increases by less than a factor of n)
    deceased_delta = (source_model_df["deceased"].iloc[-1] -
                      follow_up_model_df["deceased"].iloc[-1])

    # Does this cause an increase?
    assert 0 > deceased_delta

    # Is the magnitude of the effect a factor less than n?
    assert abs(source_model.population_size*n) > abs(deceased_delta)

    # Check 2) (Recovered increases by more than 1/n)
    # NOTE: error in Table 1 of the original paper; row 2 of MR2 should say
    # decreases the number recovered by more than 1/n, not increases. It
    # doesn't make sense for an increase in mortality probability to cause an
    # increase in recoveries.
    recovered_delta = (source_model_df["recovered"].iloc[-1] -
                       follow_up_model_df["recovered"].iloc[-1])

    # Does this cause a decrease?
    assert 0 < recovered_delta

    # Is the magnitude of the effect greater than t 1/n?
    assert abs(1 / n) < abs(recovered_delta)
