# import random
# import numpy as np


# # Define the fitness function (replace with your actual fitness function)
# def fitness_function(x):
#     return np.sum(x**2)


# def ssa_update(swarm, leader, follower, c1, c2, r1, r2):
#     new_position = leader + r1 * (follower - leader)
#     return new_position


# def pso_update(swarm, swarm_best, global_best, c1, c2, r1, r2):
#     velocity = swarm - swarm_best
#     velocity *= c1 * r1
#     velocity += c2 * r2 * (global_best - swarm)
#     swarm += velocity
#     return swarm


# def hybrid_ssa_pso_algorithm(pop_size, max_iterations, accuracy_threshold):
#     # Initialize parameters
#     swarm = np.random.rand(pop_size, 10) * 100
#     swarm_best = swarm.copy()
#     global_best = np.min(fitness_function(swarm_best))

#     iteration = 0

#     while iteration < max_iterations and global_best >= accuracy_threshold:
#         # Update the position of each particle using SSA
#         for i in range(pop_size):
#             leader = swarm[random.randint(0, pop_size - 1)]
#             follower = swarm[i]
#             r1, r2 = random.random(), random.random()
#             swarm[i] = ssa_update(follower, leader, follower, 1.5, 1.5, r1, r2)

#         # Update the position of each particle using PSO
#         for i in range(pop_size):
#             r1, r2 = random.random(), random.random()
#             swarm[i] = pso_update(
#                 swarm[i], swarm_best[i], global_best, 0.75, 1.5, r1, r2
#             )
#             fitness = fitness_function(swarm[i])
#             if fitness < swarm_best[i]:
#                 swarm_best[i] = swarm[i]
#             if fitness < global_best:
#                 global_best = fitness

#         iteration += 1

#     return global_best


# # Example usage:
# best_fitness = hybrid_ssa_pso_algorithm(
#     pop_size=10, max_iterations=10, accuracy_threshold=1e-4
# )
# print("Optimal Fitness:", best_fitness)


# import numpy as np
# import random

# lower_bound = 0
# upper_bound = 10


# class HybridSSAPSO:
#     def __init__(self, swarm_size, max_iterations, objective_function):
#         self.swarm_size = swarm_size
#         self.max_iterations = max_iterations
#         self.objective_function = objective_function

#         # Initialize the swarm
#         self.swarm = np.random.rand(swarm_size) * 10
#         self.swarm_velocity = np.random.rand(swarm_size) * 0
#         self.fitness = np.zeros(swarm_size, dtype=float)
#         self.swarm_best = self.swarm.copy()
#         self.global_best = self.swarm_best[0]
#         self.global_best_fitness = self.fitness[0]

#     def update(self):
#         # SSA Leader Phase
#         for i in range(self.swarm_size):
#             leader = self.global_best
#             follower = self.swarm_best[i]
#             c1 = 2 * np.exp(
#                 -4 * (i / self.max_iterations) ** 2
#             )  # Adjust c1 based on iteration
#             c2 = random.random()
#             c3 = random.random()
#             if c3 >= 0.5:
#                 new_position = leader + c1 * (
#                     (upper_bound - lower_bound) * c2 + lower_bound
#                 )
#             else:
#                 new_position = leader - c1 * (
#                     (upper_bound - lower_bound) * c2 + lower_bound
#                 )
#             self.swarm[i] = np.clip(new_position, lower_bound, upper_bound)

#         # SSA Followers Phase
#         for i in range(1, self.swarm_size):
#             t = i  # Iteration
#             velocity = self.swarm[i] - self.swarm[i - 1]
#             acceleration = 1 / 2 * t**2
#             self.swarm[i] = (
#                 1 / 2 * (self.swarm[i] + self.swarm[i - 1]) + velocity * acceleration
#             )

#         # PSO Phase
#         for i in range(self.swarm_size):
#             # Calculate the velocity
#             c1 = 5
#             c2 = 15
#             r1 = random.random()
#             r2 = random.random

#             velocity = self.swarm[i] - self.swarm_best[i]
#             velocity *= c1 * r1
#             velocity += c2 * r2 * (self.global_best - self.swarm[i])
#             self.swarm_velocity[i] = self.swarm_velocity[i] + velocity

#             # Update the position
#             new_position = self.swarm[i] + velocity
#             self.swarm[i] = np.clip(new_position, lower_bound, upper_bound)

#         # Evaluate the fitness of the swarm
#         self.fitness_temp = self.objective_function(self.swarm)

#         # Update the swarm_best and global_best
#         for i in range(self.swarm_size):
#             if self.fitness_temp[i] > self.fitness[i]:
#                 self.swarm_best[i] = self.swarm[i].copy()
#             if self.fitness_temp[i] > self.global_best_fitness:
#                 self.global_best_fitness = self.fitness_temp[i]
#                 self.global_best = self.swarm[i].copy()

#         self.fitness = self.fitness_temp.copy()

#     def run(self):
#         for i in range(self.max_iterations):
#             self.update()
#             print(self.global_best)
#         return self.global_best


# def sphere_function(x):
#     return np.square(x)


# swarm_size = 30
# max_iterations = 1

# # Create a new HybridSSAPSO object
# hybrid_ssapso = HybridSSAPSO(swarm_size, max_iterations, sphere_function)

# # Run the algorithm
# global_best = hybrid_ssapso.run()

# # Print the global best solution
# print(global_best)


# import numpy as np
# import random

# lower_bound = 0
# upper_bound = 10


# class HybridSSAPSO:
#     def __init__(self, swarm_size, max_iterations, objective_function):
#         self.swarm_size = swarm_size
#         self.max_iterations = max_iterations
#         self.objective_function = objective_function

#         # Initialize the swarm within the specified bounds
#         self.swarm = np.random.uniform(lower_bound, upper_bound, swarm_size)
#         self.swarm_velocity = np.random.uniform(
#             -1, 1, swarm_size
#         )  # Initialize velocities within [-1, 1]
#         self.fitness = np.zeros(swarm_size, dtype=float)
#         self.swarm_best = self.swarm.copy()
#         self.global_best_index = np.argmax(self.fitness)
#         self.global_best = self.swarm_best[self.global_best_index]
#         self.global_best_fitness = self.fitness[self.global_best_index]

#         print(1, self.swarm)
#         print(11, self.swarm_velocity)
#         print(111, self.global_best)
#         print(1111, self.global_best_fitness)

#     def update(self):
#         # SSA Leader Phase
#         for i in range(self.swarm_size):
#             leader = self.global_best
#             follower = self.swarm_best[i]
#             c1 = 2 * np.exp(-4 * (i / self.max_iterations) ** 2)
#             c2 = random.random()
#             c3 = random.random()

#             if c3 >= 0.5:
#                 new_position = leader + c1 * (upper_bound - lower_bound) * c2
#             else:
#                 new_position = leader - c1 * (upper_bound - lower_bound) * c2

#             # Update the position
#             # self.swarm[i] = np.clip(new_position, lower_bound, upper_bound)

#         # SSA Followers Phase
#         for i in range(1, self.swarm_size):
#             t = i  # Iteration
#             velocity = self.swarm[i] - self.swarm[i - 1]
#             acceleration = 1 / 2 * t**2
#             self.swarm[i] = (
#                 1 / 2 * (self.swarm[i] + self.swarm[i - 1]) + velocity * acceleration
#             )

#         # PSO update
#         for i in range(self.swarm_size):
#             c1 = 1  # Adjust according to the problem
#             c2 = 1  # Adjust according to the problem
#             r1 = random.random()
#             r2 = random.random()

#             velocity = (
#                 self.swarm_velocity[i]
#                 + c1 * r1 * (self.swarm_best[i] - self.swarm[i])
#                 + c2 * r2 * (self.global_best - self.swarm[i])
#             )
#             velocity = np.clip(velocity, -1, 1)
#             self.swarm_velocity[i] = velocity

#             # Update the position
#             new_position = self.swarm[i] + velocity
#             self.swarm[i] = np.clip(new_position, lower_bound, upper_bound)

#         # Evaluate the fitness of the swarm
#         self.fitness_temp = self.objective_function(self.swarm)

#         # Update the swarm_best and global_best
#         for i in range(self.swarm_size):
#             if self.fitness_temp[i] > self.fitness[i]:
#                 self.swarm_best[i] = self.swarm[i].copy()
#             if self.fitness_temp[i] > self.global_best_fitness:
#                 self.global_best_fitness = self.fitness_temp[i]
#                 self.global_best = self.swarm[i].copy()

#         self.fitness = self.fitness_temp.copy()

#     def run(self):
#         for i in range(self.max_iterations):
#             self.update()
#         return self.global_best


# def sphere_function(x):
#     return x**2


# swarm_size = 3
# max_iterations = 1

# hybrid_ssapso = HybridSSAPSO(swarm_size, max_iterations, sphere_function)
# global_best = hybrid_ssapso.run()
# print("Global Best:", global_best)


import numpy as np
import random

lower_bound = 1
upper_bound = 5  # num_VMs


class HybridSSAPSO:
    def __init__(
        self, swarm_size, max_iterations, objective_function, num_tasks, num_vms
    ):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.objective_function = objective_function
        self.num_tasks = num_tasks
        self.num_vms = num_vms

        # Initialize the swarm within the specified bounds
        self.swarm = np.round(
            np.random.uniform(1, self.num_vms, (swarm_size, num_tasks))
        )
        # print(self.swarm)
        self.swarm_velocity = np.random.uniform(-1, 1, (swarm_size, num_tasks))
        self.fitness = np.zeros(swarm_size, dtype=float)
        self.swarm_best = self.swarm.copy()
        self.global_best_index = np.argmin(self.fitness)
        self.global_best = self.swarm_best[self.global_best_index]
        self.global_best_fitness = self.fitness[self.global_best_index]

    def update(self):
        for i in range(self.swarm_size):
            # SSA Leader Phase
            for j in range(self.num_tasks):
                leader = self.global_best
                follower = self.swarm_best[i]
                c1 = 2 * np.exp(-4 * (i / self.max_iterations) ** 2)
                c2 = random.random()
                c3 = random.random()

                if c3 >= 0.5:
                    new_position = leader[j] + c1 * (upper_bound - lower_bound) * c2
                else:
                    new_position = leader[j] - c1 * (upper_bound - lower_bound) * c2

                # Update the position for the jth task
                self.swarm[i, j] = np.round(np.clip(new_position, 1, self.num_vms))

            # SSA Followers Phase
            for j in range(1, self.num_tasks):
                t = j  # Iteration
                velocity = self.swarm[i, j] - self.swarm[i, j - 1]
                acceleration = 1 / 2 * t**2
                self.swarm[i, j] = (
                    1 / 2 * (self.swarm[i, j] + self.swarm[i, j - 1])
                    + velocity * acceleration
                )

            # PSO update
            for j in range(self.num_tasks):
                c1 = 1  # Adjust according to the problem
                c2 = 1  # Adjust according to the problem
                r1 = random.random()
                r2 = random.random()

                velocity = (
                    self.swarm_velocity[i, j]
                    + c1 * r1 * (self.swarm_best[i, j] - self.swarm[i, j])
                    + c2 * r2 * (self.global_best[j] - self.swarm[i, j])
                )
                velocity = np.clip(velocity, -1, 1)
                self.swarm_velocity[i, j] = velocity

                # Update the position for the jth task
                new_position = self.swarm[i, j] + velocity
                self.swarm[i, j] = np.round(np.clip(new_position, 1, self.num_vms))

        # print(2, self.swarm)
        # Evaluate the fitness of the swarm
        self.fitness_temp = np.array(
            [
                self.objective_function(self.swarm[i], self.num_vms)
                for i in range(self.swarm_size)
            ]
        )

        # Update the swarm_best and global_best
        for i in range(self.swarm_size):
            if self.fitness_temp[i] < self.fitness[i]:
                self.swarm_best[i] = self.swarm[i].copy()
            if self.fitness_temp[i] < self.global_best_fitness:
                self.global_best_fitness = self.fitness_temp[i]
                self.global_best = self.swarm[i].copy()

        self.fitness = self.fitness_temp.copy()

    def run(self):
        for i in range(self.max_iterations):
            self.update()
        return self.global_best


def objective_function(task_assignments, num_vms):
    # makespans = []
    print(task_assignments)
    # for assignment in task_assignments:
    #     # print(assignment)
    #     makespan = 0
    #     # for j in range(num_vms):
    #     #     busy_time = 0
    #     #     for i in range(1):
    #     #         busy_time += assignment[i] * task_execution_time(i, j)
    #     #     if busy_time > makespan:
    #     #         makespan = busy_time
    #     # makespans.append(makespan)
    #     # busy_time = 0
    #     # for j in range(num_vms):
    #     #     busy_time += assignment[i] * task_execution_time(i, j)
    #     #     if busy_time > makespan:
    #     #         makespan = busy_time
    #     makespans.append(makespan)
    # return np.min(makespans)
    busy_times = np.zeros(num_vms + 1)
    task = 0
    for assignment in task_assignments:
        assignment = int(assignment)
        busy_times[assignment] += task_execution_time(int(task), int(assignment))
        task += 1
    return np.min(busy_times)


def task_execution_time(task_id, vm_id):
    # Your task execution time calculation logic here
    # For example, Li is the task length, and CUj is the VM capacity
    Li = task_length[task_id]
    CUj = vm_capacity[vm_id]
    return Li / CUj


# Define the number of tasks and VMs
num_tasks = 10
num_vms = 5

task_length_range = (1e3, 1e5)  # Replace with your desired integer range
vm_capacity_range = (1e2, 1e3)  # Replace with your desired integer range

# Create arrays with random integer values in the specified ranges
task_length = np.random.randint(
    task_length_range[0], task_length_range[1] + 1, num_tasks + 1
)
vm_capacity = np.random.randint(
    vm_capacity_range[0], vm_capacity_range[1] + 1, num_vms + 1
)

print(task_length)
print(vm_capacity)


swarm_size = 10
max_iterations = 20

hybrid_ssapso = HybridSSAPSO(
    swarm_size, max_iterations, objective_function, num_tasks, num_vms
)
global_best = hybrid_ssapso.run()
print("Global Best:", global_best)
