import numpy as np
import random

lower_bound = 0
upper_bound = 20


class HybridSSAPSO:
    def __init__(self, swarm_size, max_iterations, objective_function):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.objective_function = objective_function

        # Initialize the swarm
        self.swarm = np.random.rand(swarm_size) * 10
        self.swarm_velocity = np.random.rand(swarm_size) * 0
        self.fitness = np.zeros(swarm_size, dtype=float)
        self.swarm_best = self.swarm.copy()
        self.global_best = self.swarm_best[0]
        self.global_best_fitness = self.fitness[0]

        print(1, self.swarm)

    def update(self):
        # SSA update
        # SSA Leader Phase
        for i in range(self.swarm_size):
            leader = self.global_best
            follower = self.swarm_best[i]
            c1 = 2 * np.exp(
                -4 * (i / self.max_iterations) ** 2
            )  # Adjust c1 based on iteration
            c2 = random.random()
            c3 = random.random()
            if c3 >= 0.5:
                new_position = leader + c1 * (
                    (upper_bound - lower_bound) * c2 + lower_bound
                )
            else:
                new_position = leader - c1 * (
                    (upper_bound - lower_bound) * c2 + lower_bound
                )
        print(2, self.swarm)

        # SSA Followers Phase
        for i in range(1, self.swarm_size):
            t = i  # Iteration
            velocity = self.swarm[i] - self.swarm[i - 1]
            acceleration = 1 / 2 * t**2
            self.swarm[i] = (
                1 / 2 * (self.swarm[i] + self.swarm[i - 1]) + velocity * acceleration
            )

        # self.swarm[i] = np.clip(new_position, lower_bound, upper_bound)
        print(3, self.swarm)

        # PSO update
        for i in range(self.swarm_size):
            # Calculate the velocity
            c1 = 5
            c2 = 15
            r1 = random.random()
            r2 = random.random()

            velocity = self.swarm[i] - self.swarm_best[i]
            velocity *= c1 * r1
            velocity += c2 * r2 * (self.global_best - self.swarm[i])
            velocity = np.clip(velocity, lower_bound, upper_bound / 10)
            self.swarm_velocity[i] = self.swarm_velocity[i] + velocity

            # Update the position
            new_position = self.swarm[i] + velocity
            self.swarm[i] = np.clip(new_position, lower_bound, upper_bound)

        # Evaluate the fitness of the swarm
        # print(1, self.fitness)
        self.fitness_temp = self.objective_function(self.swarm)
        # print(self.fitness)
        # print(self.fitness_temp)
        # print(self.swarm)
        # print(self.swarm_best)
        # print(2, self.fitness)

        # Update the swarm_best and global_best
        for i in range(self.swarm_size):
            print(i, self.fitness_temp[i], self.fitness[i], self.global_best)
            if self.fitness_temp[i] > self.fitness[i]:
                self.swarm_best[i] = self.swarm[i].copy()
            if self.fitness_temp[i] > self.global_best_fitness:
                self.global_best_fitness = self.fitness_temp[i]
                self.global_best = self.swarm[i].copy()

        self.fitness = self.fitness_temp.copy()

    def run(self):
        for i in range(self.max_iterations):
            self.update()
            print(self.global_best)
        return self.global_best


def sphere_function(x):
    return 1 / np.square(x)


swarm_size = 30
max_iterations = 100

# Create a new HybridSSAPSO object
hybrid_ssapso = HybridSSAPSO(swarm_size, max_iterations, sphere_function)

# Run the algorithm
global_best = hybrid_ssapso.run()

# Print the global best solution
print(global_best)
