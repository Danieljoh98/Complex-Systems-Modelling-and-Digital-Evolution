"""Hypothetical network SIR epidemic with age groups and random immunisation.

Run normally for the PyCX GUI, or use --headless for one numerical run.
Age distributions and contact differences are illustrative, not empirical data.
"""

import argparse

import numpy as np

n = 1000
mean_contacts = 8      # expected mean degree, not an exact contact count
p_inf = 0.03           # infection probability per infected neighbour per step
p_rec = 0.10           # recovery probability per infected person per step
immunisation_fraction = 0.10
seed = None

age_labels = ('0-29', '30-59', '60+')
age_proportions = (0.35, 0.45, 0.20)
contact_weights = np.array([1.5, 1.0, 0.65])

S, I, R, V = range(4)  # vaccinated people are counted separately from recovered
state_labels = ('Susceptible', 'Infected', 'Recovered', 'Immunised')
state_colours = np.array(['cyan', 'red', 'green', 'blue'])


def create_network(age_groups, rng):
    """Create fixed contacts, weighted by each person's age group."""
    # Each person is a node. An undirected edge is a permanent contact.
    # Higher contact weights give a group more expected contacts. There is
    # no additional preference for contacts within the same age group.
    activity = contact_weights[age_groups]
    left, right = np.triu_indices(n, k=1)
    pair_weights = activity[left] * activity[right]
    probabilities = mean_contacts * n / 2 * pair_weights / pair_weights.sum()
    connected = rng.random(len(left)) < np.minimum(probabilities, 1)
    return np.column_stack((left[connected], right[connected]))

# TODO: Add GA-based allocation of the same vaccine budget across age groups.
# Keep random immunisation as the baseline for comparison.
def initial_states(rng):
    """Immunise random people, then infect one susceptible person."""
    states = np.full(n, S, dtype=int)
    # Reserve one unvaccinated person to introduce the infection.
    budget = min(int(n * immunisation_fraction), n - 1)
    vaccinated = rng.choice(n, size=budget, replace=False)
    states[vaccinated] = V
    patient_zero = rng.choice(np.flatnonzero(states == S))
    states[patient_zero] = I
    return states


def create_positions(age_groups):
    # Fixed positions are only for drawing; distance does not affect infection.
    layout_rng = np.random.default_rng(0)
    return np.column_stack((
        age_groups + layout_rng.uniform(-0.35, 0.35, n),
        layout_rng.uniform(0, 1, n),
    ))


def initialize():
    global rng, age_groups, edges, states, history, total_infected, positions
    rng = np.random.default_rng(seed)
    age_groups = rng.choice(len(age_labels), size=n, p=age_proportions)
    edges = create_network(age_groups, rng)
    states = initial_states(rng)
    positions = create_positions(age_groups)
    total_infected = 1
    history = [np.bincount(states, minlength=len(state_labels))]


def update():
    global total_infected
    if not np.any(states == I):
        return

    # Use the state at the start of the step for all transmission and recovery.
    infected = states == I
    infected_neighbours = np.zeros(n, dtype=int)
    left, right = edges.T
    np.add.at(infected_neighbours, left, infected[right])
    np.add.at(infected_neighbours, right, infected[left])
    infection_probability = 1 - (1 - p_inf) ** infected_neighbours
    new_infections = (states == S) & (rng.random(n) < infection_probability)
    recoveries = infected & (rng.random(n) < p_rec)
    states[new_infections] = I
    states[recoveries] = R
    total_infected += int(new_infections.sum())
    history.append(np.bincount(states, minlength=len(state_labels)))


def observe():
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.clf()
    ax = plt.subplot(1, 2, 1)
    ax.add_collection(LineCollection(
        positions[edges], colors='gray', linewidths=0.2, alpha=0.08,
    ))
    ax.scatter(positions[:, 0], positions[:, 1], c=state_colours[states], s=12)
    ax.set_xticks(range(len(age_labels)), age_labels)
    ax.set_xlabel('Age group (illustrative contact assumptions)')
    ax.set_yticks([])
    ax.set_title('Fixed contact network')

    ax = plt.subplot(1, 2, 2)
    counts = np.asarray(history)
    for state, label in enumerate(state_labels):
        ax.plot(counts[:, state], label=label, color=state_colours[state])
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('People')
    ax.set_title(f'Total infected: {total_infected} / {n}')
    ax.legend()
    plt.tight_layout()


def run_simulation(max_steps):
    """Run until the outbreak ends or the step limit is reached."""
    initialize()
    for _ in range(max_steps):
        if not np.any(states == I):
            break
        update()


def print_results():
    print(f'People: {n}; contacts: {len(edges)}; immunised: {history[0][V]}')
    degree = np.bincount(edges.ravel(), minlength=n)
    for group, label in enumerate(age_labels):
        members = age_groups == group
        average = degree[members].mean() if members.any() else 0
        print(f'Age {label}: {members.sum()} people, '
              f'mean contacts {average:.2f}')
    print(f'Total infected: {total_infected}; '
          f'peak infected: {max(count[I] for count in history)}; '
          f'steps: {len(history) - 1}')
    print('Outbreak ended' if not np.any(states == I)
          else 'Step limit reached: outbreak still active')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true',
                        help='run without the GUI and print epidemic outcomes')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--steps', type=int, default=1000,
                        help='maximum number of steps in headless mode')
    parser.add_argument('--immunisation-fraction', type=float,
                        default=immunisation_fraction)
    args = parser.parse_args()
    if not 0 <= args.immunisation_fraction <= 1:
        parser.error('--immunisation-fraction must be between 0 and 1')
    if args.steps < 1:
        parser.error('--steps must be positive')
    return args


def immunisation_coverage(value=None):
    """Fraction immunised, from 0 to 1. Save and reset to apply."""
    global immunisation_fraction
    if value is not None:
        if not 0 <= value <= 1:
            raise ValueError('Immunisation coverage must be between 0 and 1')
        immunisation_fraction = value
    return immunisation_fraction


def infection_probability(value=None):
    """Infection probability per infected neighbour per step, from 0 to 1."""
    global p_inf
    if value is not None:
        if not 0 <= value <= 1:
            raise ValueError('Infection probability must be between 0 and 1')
        p_inf = value
    return p_inf


def recovery_probability(value=None):
    """Recovery probability per infected person per step, from 0 to 1."""
    global p_rec
    if value is not None:
        if not 0 <= value <= 1:
            raise ValueError('Recovery probability must be between 0 and 1')
        p_rec = value
    return p_rec


def main():
    global seed, immunisation_fraction
    args = parse_args()
    seed = args.seed
    immunisation_fraction = args.immunisation_fraction

    if args.headless:
        run_simulation(args.steps)
        print_results()
    else:
        import pycxsimulator
        pycxsimulator.GUI(parameterSetters=[
            immunisation_coverage, infection_probability, recovery_probability,
        ]).start(func=[initialize, observe, update])


if __name__ == '__main__':
    main()
