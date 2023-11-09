import numpy as np
import random

N = 5
T = 10
lower_bound = 1
upper_bound = N  # num_VMs


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

        self.swarm_velocity = np.random.uniform(-1, 1, (swarm_size, num_tasks))
        self.fitness = np.zeros(swarm_size, dtype=float)
        self.swarm_best = self.swarm.copy()
        self.global_best_index = np.argmin(self.fitness)
        self.global_best = self.swarm_best[self.global_best_index]
        self.global_best_fitness = 10**18

    def update(self, itr):
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
                self.swarm[i, j] = bring_2d_in_range(
                    new_position, lower_bound, upper_bound
                )

            # SSA Followers Phase
            for j in range(1, self.num_tasks):
                t = j  # Iteration
                velocity = self.swarm[i, j] - self.swarm[i, j - 1]
                acceleration = 1 / 2 * t**2
                self.swarm[i, j] = (
                    1 / 2 * (self.swarm[i, j] + self.swarm[i, j - 1])
                    + velocity * acceleration
                )
                self.swarm[i, j] = bring_2d_in_range(
                    new_position, lower_bound, upper_bound
                )

            # PSO update
            for j in range(self.num_tasks):
                c1 = 0.5  # Adjust according to the problem
                c2 = 1  # Adjust according to the problem
                r1 = random.random()
                r2 = random.random()

                velocity = (
                    self.swarm_velocity[i, j]
                    + c1 * r1 * (self.swarm_best[i, j] - self.swarm[i, j])
                    + c2 * r2 * (self.global_best[j] - self.swarm[i, j])
                )
                # print(velocity)
                velocity = np.clip(velocity, -1, 1)
                # print("#", velocity)
                self.swarm_velocity[i, j] = velocity

                # Update the position for the jth task
                new_position = self.swarm[i, j] + velocity
                self.swarm[i, j] = bring_2d_in_range(
                    new_position, lower_bound, upper_bound
                )

        # Evaluate the fitness of the swarm
        self.fitness_temp = np.array(
            [
                self.objective_function(
                    self.swarm[i],
                    self.num_vms,
                )
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

        print(self.global_best)
        self.fitness = self.fitness_temp.copy()

        # OBL
        if itr % 2 == 0 or itr % 2 == 1:
            for i in range(self.swarm_size):
                rand = random.random()
                if rand > 0.5:
                    self.swarm[i] = [N - x for x in self.swarm[i]]

    def run(self):
        for i in range(self.max_iterations):
            self.update(i)
        return [self.global_best, self.global_best_fitness]


def bring_2d_in_range(new_position, lower_bound, upper_bound):
    if new_position < lower_bound or new_position > upper_bound:
        return random.randint(lower_bound, upper_bound)
    range_size = upper_bound - lower_bound
    result = lower_bound + (new_position - lower_bound) % (range_size + 1)
    return int(result)


def objective_function(task_assignments, num_vms):
    vm_capacity_temp = vm_capacity.copy()  # Create a copy of vm_capacity
    busy_times = np.zeros(num_vms + 1)
    task = 1
    for assignment in task_assignments:
        assignment = int(assignment)
        busy_times[assignment] += task_execution_time(
            int(task), int(assignment), vm_capacity_temp
        )
        task += 1
        # print("Before calculation: busy_times[assignment] =", busy_times[assignment])
        vm_capacity_temp[assignment] /= normalize_value(busy_times[assignment])

    makespan = np.max(busy_times)

    # Calculate resource utilization
    ru = 0
    for vm_id in range(num_vms):
        ru += busy_times[vm_id + 1] / (makespan * num_vms)

    # Combine the makespan and resource utilization using weights alpha_1 and alpha_3
    alpha_1 = 0.7  # Adjust according to the problem
    alpha_2 = 0.3  # Adjust according to the problem
    objective_value = alpha_1 * makespan + alpha_2 * (1 - ru)
    return objective_value


def task_execution_time(task_id, vm_id, vm_capacity_temp):
    # Your task execution time calculation logic here
    # For example, Li is the task length, and CUj is the VM capacity
    Li = task_length[task_id]
    CUj = vm_capacity_temp[vm_id]
    return Li / CUj


def normalize_value(value, min_value=1, max_value=400):
    # Ensure the value is within the original range
    if value < min_value:
        value = min_value
    elif value > max_value:
        value = max_value

    # Calculate the normalized value in the target range
    normalized_value = 1 + (value - min_value) * (1 / (max_value - min_value))

    return normalized_value


# Define the number of tasks and VMs
num_tasks = T
num_vms = N

task_length_range = (1e8, 2 * 1e9)  # Replace with your desired integer range
vm_capacity_range = (1e7, 1e8)  # Replace with your desired integer range

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


def EvaluationMetrics():
    vm_capacity_temp = vm_capacity.copy()  # Create a copy of vm_capacity
    busy_times = np.zeros(num_vms + 1)
    task = 1
    throughput = 0
    for assignment in ans[0]:
        assignment = int(assignment)
        busy_times[assignment] += task_execution_time(
            int(task), int(assignment), vm_capacity_temp
        )
        throughput += busy_times[assignment]
        task += 1
        vm_capacity_temp[assignment] /= normalize_value(busy_times[assignment])
    # print(busy_times)
    makespan = np.max(busy_times)

    # Calculate resource utilization
    ru = 0
    for vm_id in range(num_vms):
        ru += busy_times[vm_id + 1] / (makespan * num_vms)

    return [throughput, ru]


print("Global Best:", ans[0])
print("Global Best Fitness", ans[1])

em = EvaluationMetrics()
print("Throughput", em[0])
print("Resource Utilization", em[1])
