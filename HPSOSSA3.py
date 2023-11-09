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
        self.global_best_fitness = 10**18

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

        # print("##", self.fitness_temp)

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
        return [self.global_best, self.global_best_fitness]


def objective_function(task_assignments, num_vms):
    # makespans = []
    # print(task_assignments)
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
    # print(busy_times)
    return np.max(busy_times)


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


swarm_size = 20
max_iterations = 20

hybrid_ssapso = HybridSSAPSO(
    swarm_size, max_iterations, objective_function, num_tasks, num_vms
)
ans = hybrid_ssapso.run()

print("Global Best:", ans[0])
print("Global Best Fitness", ans[1])

throughput = 0
tasks_num = 1
for t in ans[0]:
    throughput += task_execution_time(int(tasks_num), int(t))
    tasks_num += 1
print("Throughput", throughput)
