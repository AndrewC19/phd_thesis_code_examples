"""
Metamorphic relations for the Spanish Influenza model as presented by Pullum
and Ozmen in their 2012 paper entitled: Early Results from Metamorphic Testing
of Epidemiological Models.
"""

import pytest
import numpy as np

from unittest import TestCase
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
    source_inputs = random_test_case()
    source_model = SpanishInfluenzaEBM(**source_inputs)
    source_model_df = source_model.solve(plot=False)

    n = np.random.uniform(0, 0.999)
    follow_up_inputs = source_inputs.copy()
    follow_up_inputs["mortality_prob"] *= n
    follow_up_model = SpanishInfluenzaEBM(**follow_up_inputs)
    follow_up_model_df = follow_up_model.solve(plot=False)

    # Check 1) (Deceased decreases by less than a factor of n)
    deceased_delta = (follow_up_model_df["deceased"].iloc[-1] -
                      source_model_df["deceased"].iloc[-1])

    assert deceased_delta < source_model_df["deceased"].iloc[-1]*n

    # Check 2) (Recovered decreases by at least a factor of 1/n)
    recovered_delta = (follow_up_model_df["recovered"].iloc[-1] -
                       source_model_df["recovered"].iloc[-1])
    assert recovered_delta < source_model_df["deceased"].iloc[-1]/n
