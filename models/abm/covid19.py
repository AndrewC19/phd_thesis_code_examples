import covasim as cv
import matplotlib.pyplot as plt

# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.size": 12,
#         "text.usetex": True,
#         "text.latex.preamble": r"\usepackage{libertine}"
#     }
# )


def covid19(scaling_factor: int = 10000,
            pars_to_change: dict = None):
    """Run Covasim with the parameters outlined in the thesis.
    N.B. A scaling factor of 100 is used by default. This massively reduces the
    simulation time at the cost of model fidelity."""
    sim_pars = {"location": "usa",
                "pop_type": "hybrid",
                "pop_infected": 1000,
                "pop_size": 103267000 / scaling_factor,
                "n_days": 200,
                "pop_scale": scaling_factor}
    if pars_to_change:
        sim_pars.update(pars_to_change)
    sim = cv.Sim(pars=sim_pars)
    return sim


def run_covid19(sim: cv.Sim,
                plot: bool = False,
                out_path: str = "covid19_abm.pdf"):

    sim.run()
    if plot:
        plot_results(sim.results, out_path)
    return sim


def plot_results(sim_results, out_path):
    """Plot results in the same format as the Spanish Influenza EBM."""
    xs = list(range(201))
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 7))
    ax[0, 0].plot(xs, sim_results["cum_infections"], color='blue',
                  linewidth=1)
    ax[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
    ax[0, 0].set_ylim(0, 9e7)
    ax[0, 0].set_ylabel("Total Infected")

    ax[0, 1].plot(xs, sim_results["n_susceptible"], color='blue',
                  linewidth=1)
    ax[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(1, 2))
    ax[0, 1].set_ylim(0, 12e7)
    ax[0, 1].set_ylabel("Susceptible")

    ax[1, 0].plot(xs, sim_results["n_exposed"], color='blue',
                  linewidth=1)
    ax[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
    ax[1, 0].set_ylim(0, 6e7)
    ax[1, 0].set_ylabel("Incubating")

    ax[1, 1].plot(xs, sim_results["n_infectious"], color='blue',
                  linewidth=1)
    ax[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
    ax[1, 1].set_ylim(0, 4e7)
    ax[1, 1].set_ylabel("Infectious")

    ax[2, 0].plot(xs, sim_results["n_dead"], color='blue',
                  linewidth=1)
    ax[2, 0].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
    ax[2, 0].set_ylim(0, 7e5)
    ax[2, 0].set_ylabel("Deceased")
    ax[2, 0].set_xlabel("Days")

    ax[2, 1].plot(xs, sim_results["n_recovered"], color='blue',
                  linewidth=1)
    ax[2, 1].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
    ax[2, 1].set_ylim(0, 9e7)
    ax[2, 1].set_ylabel("Recovered")
    ax[2, 1].set_xlabel("Days")
    plt.suptitle("COVID-19 in USA")
    plt.tight_layout()
    plt.show()
    if out_path:
        fig.savefig(out_path, format='pdf', dpi=300)


if __name__ == "__main__":
    sim = covid19(scaling_factor=10000)
    executed_sim = run_covid19(sim, plot=True)

    follow_up_sim = covid19(scaling_factor=10000)
    follow_up_sim.pars["prognoses"]["death_probs"] *= 0.5
    executed_follow_up_sim = run_covid19(follow_up_sim, plot=True)
