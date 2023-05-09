import pandas as pd
import numpy as np
from pprint import pprint

RESULTS_CSV_PATH = "metamorphic_testing_data.csv"


def mr_stats():
    results_df = pd.read_csv(RESULTS_CSV_PATH)
    results_dict = {}
    for mr in results_df["relation"].unique():
        mr_df = results_df.loc[results_df["relation"] == mr]
        passes = mr_df["test_pass"].sum()
        total_ex_time = mr_df["time"].sum()
        min_ex_time = mr_df["time"].min()
        max_ex_time = mr_df["time"].max()
        mean_ex_time = mr_df["time"].mean()
        fails = len(mr_df) - passes
        results_dict[mr] = {"passes": passes,
                            "fails": fails,
                            "pass_rate": passes/len(mr_df),
                            "total_time": total_ex_time,
                            "min_ex_time": min_ex_time,
                            "max_ex_time": max_ex_time,
                            "mean_ex_time": mean_ex_time}
    del results_dict["source"]
    return results_dict


if __name__ == "__main__":
    stats = mr_stats()
    pprint(stats)
