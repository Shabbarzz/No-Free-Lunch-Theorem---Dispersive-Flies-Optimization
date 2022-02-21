
# Possibly the best Machine Learning library ever
import numpy as np

# Let's print the output nicely
np.set_printoptions(precision=3, suppress=True)

# Population Size
size = 100

# How regularily the flies are dispersed
disturbance_threshold = 0.01

# The target solution we are aiming to optimise towards
target_solution = np.array([0.0, 1.0, -1.0, 1.0, 0.0])

# Upper and lower bounds for initialisation steps
lower = -1.0
upper = 1.0

# Create a population from the upper and lower bounds - note a uniform random distribution is used
population = np.array([np.random.uniform(lower, upper, len(target_solution)) for _ in range(size)])

# Because the algorithm is stochastic sometimes it is nice to record the best solution
all_time_best = None
all_time_best_score = np.finfo(np.float32).max

# An empty vessel to contain each flies best neighbour
best_neighbour = np.zeros_like(population[0])

# How many iterations of optimisation we want to compute
iteration_amount = 100
for _ in range(iteration_amount):

    # Compute the fitnesses for each fly by calculating the l2 (euclidean) loss from the target vector
    fitnesses = np.zeros(len(population))
    for i in range(len(population)):
        fitnesses[i] = np.linalg.norm(target_solution - population[i])

    # Which flies index has the lowest loss?
    swarms_best_index = np.argmin(fitnesses)

    # Get best fly
    swarms_best = population[swarms_best_index]

    # Record best fly of all time
    if np.amin(fitnesses) <= all_time_best_score:
        all_time_best_score = np.amin(fitnesses)
        all_time_best = swarms_best

    # All the random dice rolls we will make for every 'd' in each member of the population
    r = np.random.uniform(0.0, 1.0, population.shape)

    # For each fly in the swarm
    for i, p in enumerate(population):

        # Get the neighbours indices - note how we turn the list into a circular buffer
        left = (i - 1) if i is not 0 else len(population) - 1
        right = (i + 1) if i is not (len(population) - 1) else 0

        # Here is the best scoring neighbouring fly
        best_neighbour = population[left] if fitnesses[left] < fitnesses[right] else population[right]

        # For each element comprising the fly
        for x in range(len(p)):

            # If the roll computed earlier is lower than the threshold, re-init the fly, else, update
            # fly to best neighbour and move it a random amount towards the swarms best fly.
            if r[i][x] < disturbance_threshold:
                p[x] = np.random.uniform(lower, upper)
            else:
                update = swarms_best[x] - best_neighbour[x]
                p[x] = best_neighbour[x] + np.random.uniform(0.0, 1.0) * update

# Get the final fitnesses with l2 diff
fitnesses = np.zeros(len(population))
for i in range(len(population)):
    fitnesses[i] = np.linalg.norm(target_solution - population[i])

# Get the best fly
swarms_best_index = np.argmin(fitnesses)
swarms_best = population[swarms_best_index]

# Print results
print('target:', target_solution)
print('best:  ', all_time_best)
print('diff:  ', np.abs(all_time_best - target_solution))

# -------------------------------------------
# Example output:
#
# target: [ 0.    1.    -1.     1.     0.   ]
# best:   [ 0.    0.992 -0.967  0.992 -0.   ]
# diff:   [ 0.    0.008  0.033  0.008  0.   ]
view rawdfo.py hosted with ❤ by GitHub
