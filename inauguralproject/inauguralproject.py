# %% [markdown]
# # Inaugural Project

# %% [markdown]
# > **Note the following:** 
# > 1. This is an example of how to structure your **inaugural project**.
# > 1. Remember the general advice on structuring and commenting your code
# > 1. The `inauguralproject.py` file includes a function which can be used multiple times in this notebook.

# %% [markdown]
# Imports and set magics:

# %%
import numpy as np

# autoreload modules when code is run. Otherwise, python will not see recent changes. 
%reload_ext autoreload
%autoreload 2



# %% [markdown]
# # Question 1

# %% [markdown]
# Code is explained throughout.
# Evereything within the blue area is pareto improvements.

# %%
# code for solving the model (remember documentation and comments)
import matplotlib.pyplot as plt

# Constants
alpha = 1/3
beta = 2/3
w1A = 0.8
w2A = 0.3
w1B = 1 - w1A
w2B = 1 - w2A
N = 75

# Utility functions
def uA(x1A, x2A):
    return x1A ** alpha * (x2A ** (1 - alpha))

def uB(x1B, x2B):
    return (x1B * beta) * (x2B * (1 - beta))

# Initial utilities
uA_initial = uA(w1A, w2A)
uB_initial = uB(w1B, w2B)

# Edgeworth box setup
x1A_vals = np.linspace(0, 1, N)
x2A_vals = np.linspace(0, 1, N)
x1A_mesh, x2A_mesh = np.meshgrid(x1A_vals, x2A_vals)

# Find the pairs that satisfy the utility conditions
pareto_improvements = np.zeros(x1A_mesh.shape)
for i in range(N):
    for j in range(N):
        if uA(x1A_mesh[i, j], x2A_mesh[i, j]) >= uA_initial and uB(1 - x1A_mesh[i, j], 1 - x2A_mesh[i, j]) >= uB_initial:
            pareto_improvements[i, j] = 1

# Plot
plt.figure(figsize=(6, 6))
plt.contourf(x1A_mesh, x2A_mesh, pareto_improvements, cmap='Blues', levels=[0.5, 1], alpha=0.5)
plt.plot(w1A, w2A, 'ro')  # Endowment point
plt.xlabel('$x_1^A$')
plt.ylabel('$x_2^A$')
plt.title('Pareto improvements in the Edgeworth Box')
plt.grid(True)
plt.show()

# %% [markdown]
# # Question 2

# %% [markdown]
# We weren't able to import the exchange economy class, so we pasted it instead and used it to calculate the market clearing errors.

# %%
from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):
        # Parameters
        par = self.par = SimpleNamespace()

        # Preferences
        par.alpha = 1/3
        par.beta = 2/3

        # Endowments
        par.w1A = 0.8
        par.w2A = 0.3

    def utility_A(self, x1A, x2A):
        # Utility function for consumer A
        return (x1A * self.par.alpha) * (x2A * (1 - self.par.alpha))

    def utility_B(self, x1B, x2B):
        # Utility function for consumer B
        return (x1B * self.par.beta) * (x2B * (1 - self.par.beta))

    def demand_A(self, p1, p2=1):
        # Demand function for consumer A
        I_A = self.par.w1A * p1 + self.par.w2A * p2
        x1A = self.par.alpha * I_A / p1
        x2A = (1 - self.par.alpha) * I_A / p2
        return x1A, x2A

    def demand_B(self, p1, p2=1):
        # Demand function for consumer B
        I_B = (1 - self.par.w1A) * p1 + (1 - self.par.w2A) * p2
        x1B = self.par.beta * I_B / p1
        x2B = (1 - self.par.beta) * I_B / p2
        return x1B, x2B

    def check_market_clearing(self, p1):
        # Check market clearing conditions
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        eps1 = x1A - self.par.w1A + x1B - (1 - self.par.w1A)
        eps2 = x2A - self.par.w2A + x2B - (1 - self.par.w2A)
        return eps1, eps2

# Now, use the model to calculate market clearing errors for a range of prices p1
model = ExchangeEconomyClass()
N = 75
p1_values = np.linspace(0.5, 2.5, N)
market_clearing_errors = []

for p1 in p1_values:
    eps1, eps2 = model.check_market_clearing(p1)
    market_clearing_errors.append((p1, eps1, eps2))

# Display the results
print("p1, Market clearing error for good 1, Market clearing error for good 2")
for p1, eps1, eps2 in market_clearing_errors:
    print(f'{p1:.2f}, {eps1:.4f}, {eps2:.4f}')

# %% [markdown]
# # Question 3

# %% [markdown]
# We find that the market-clearing price p1 is approximately: 0.93

# %%
# Initialize variables to store the best (smallest) error and corresponding p1
best_error = float('inf')  # Set initial best error to be infinitely large
best_p1 = None  # Initialize best p1 as None

# Go through each price and its errors
for p1, eps1, eps2 in market_clearing_errors:
    total_error = abs(eps1) + abs(eps2)  # Combine the absolute errors
    if total_error < best_error:
        best_error = total_error
        best_p1 = p1

# Print the market-clearing price
print(f'The market-clearing price p1 is approximately: {best_p1:.2f}')

# %% [markdown]
# # Question 4a

# %% [markdown]
# We find the best allocation under given constraints

# %%
# Define the price range for P1
N = 75
p1_values = np.linspace(0.5, 2.5, N)
max_utility = -np.inf  # Start with a very low utility value
best_p1 = None  # Best price is unknown initially
best_allocation = None  # Best allocation is unknown initially

for p1 in p1_values:
    x1B, x2B = model.demand_B(p1)
    # Consumer A gets whatever is left after B's consumption
    x1A = 1 - x1B
    x2A = 1 - x2B
    # Calculate A's utility for this allocation
    utility = model.utility_A(x1A, x2A)
    # Check if this utility is better than the best one found so far
    if utility > max_utility:
        max_utility = utility
        best_p1 = p1
        best_allocation = (x1A, x2A)

# Display the results
print(f'Best price p1 for A to choose: {best_p1:.2f}')
print(f'Best allocation for A (x1A, x2A): ({best_allocation[0]:.2f}, {best_allocation[1]:.2f})')
print(f'Maximized utility for A: {max_utility:.4f}')

# %% [markdown]
# # Question 4b

# %% [markdown]
# insert code and explain 

# %%
from scipy.optimize import minimize

# Parameters
alpha = 1 / 3
beta = 2 / 3
omega_A1 = 0.8
omega_A2 = 0.3
omega_B1 = 1 - omega_A1
omega_B2 = 1 - omega_A2

def utility_A(x1, x2):
    return (x1 ** alpha) * (x2 ** (1 - alpha))

def utility_B(x1, x2):
    return (x1 ** beta) * (x2 ** (1 - beta))

# Demand functions
def demand_A(p1, p2, omega_A1, omega_A2, alpha):
    x1_A = alpha * (p1 * omega_A1 + p2 * omega_A2) / p1
    x2_A = (1 - alpha) * (p1 * omega_A1 + p2 * omega_A2) / p2
    return x1_A, x2_A

def demand_B(p1, p2, omega_B1, omega_B2, beta):
    x1_B = beta * (p1 * omega_B1 + p2 * omega_B2) / p1
    x2_B = (1 - beta) * (p1 * omega_B1 + p2 * omega_B2) / p2
    return x1_B, x2_B

# Objective function to maximize A's utility
def objective(p1):
    x1_A, x2_A = demand_A(p1, 1, omega_A1, omega_A2, alpha)
    return -utility_A(x1_A, x2_A)

# Bounds for the price p1
bounds = [(0.01, 10)]

# Initial guess for the price p1
p1_initial = 1

# Perform the optimization
result = minimize(objective, p1_initial, bounds=bounds)
optimal_p1_continuous = result.x[0]
max_utility_A_continuous = -result.fun

print(f'Optimal p1 (continuous): {optimal_p1_continuous}')
print(f'Maximum utility for A (continuous): {max_utility_A_continuous}')

# Corresponding demands
optimal_demand_A = demand_A(optimal_p1_continuous, 1, omega_A1, omega_A2, alpha)
optimal_demand_B = demand_B(optimal_p1_continuous, 1, omega_B1, omega_B2, beta)

print(f'Optimal demand for A: x1_A = {optimal_demand_A[0]}, x2_A = {optimal_demand_A[1]}')
print(f'Optimal demand for B: x1_B = {optimal_demand_B[0]}, x2_B = {optimal_demand_B[1]}')


# %% [markdown]
# # Question 5a

# %% [markdown]
# We find the best allocation under given constraints

# %%
# Parameters
alpha = 1 / 3
beta = 2 / 3
omega_A1 = 0.8
omega_A2 = 0.3
omega_B1 = 1 - omega_A1
omega_B2 = 1 - omega_A2
N = 75

def utility_A(x1, x2):
    return (x1 ** alpha) * (x2 ** (1 - alpha))

def utility_B(x1, x2):
    return (x1 ** beta) * (x2 ** (1 - beta))

uA_initial = utility_A(omega_A1, omega_A2)
uB_initial = utility_B(omega_B1, omega_B2)

x1_A_values = np.linspace(0, 1, N)
x2_A_values = np.linspace(0, 1, N)
C_set = []

# Find the set C
for x1_A in x1_A_values:
    for x2_A in x2_A_values:
        x1_B = 1 - x1_A
        x2_B = 1 - x2_A
        if utility_A(x1_A, x2_A) >= uA_initial and utility_B(x1_B, x2_B) >= uB_initial:
            C_set.append((x1_A, x2_A))

C_set = np.array(C_set)

# Find the allocation in set C that maximizes A's utility
max_utility_A = -np.inf
optimal_allocation_A = (0, 0)

for allocation in C_set:
    x1_A, x2_A = allocation
    current_utility_A = utility_A(x1_A, x2_A)
    if current_utility_A > max_utility_A:
        max_utility_A = current_utility_A
        optimal_allocation_A = (x1_A, x2_A)

# Results
print(f'Optimal allocation for A within set C: x1_A = {optimal_allocation_A[0]}, x2_A = {optimal_allocation_A[1]}')
print(f'Maximum utility for A within set C: {max_utility_A}')



# %% [markdown]
# # Question 5b

# %% [markdown]
# We solve the problem with given constraints ensuring that consumer B is not worse off

# %%
# Parameters
alpha = 1 / 3
beta = 2 / 3
omega_A1 = 0.8
omega_A2 = 0.3
omega_B1 = 1 - omega_A1
omega_B2 = 1 - omega_A2

def utility_A(x1, x2):
    return (x1 ** alpha) * (x2 ** (1 - alpha))

def utility_B(x1, x2):
    return (x1 ** beta) * (x2 ** (1 - beta))

uB_initial = utility_B(omega_B1, omega_B2)

# Objective function to maximize A's utility
def objective(x):
    x1_A, x2_A = x
    return -utility_A(x1_A, x2_A)

# Constraint to ensure B's utility is at least the initial utility
def constraint(x):
    x1_A, x2_A = x
    x1_B = 1 - x1_A
    x2_B = 1 - x2_A
    return utility_B(x1_B, x2_B) - uB_initial

# Bounds for the variables
bounds = [(0, 1), (0, 1)]

# Initial guess
x0 = [omega_A1, omega_A2]

# Constraint dictionary
cons = {'type': 'ineq', 'fun': constraint}

# Perform the optimization
result = minimize(objective, x0, bounds=bounds, constraints=cons)
optimal_allocation_A_unrestricted = result.x
max_utility_A_unrestricted = -result.fun

print(f'Optimal allocation for A without restrictions: x1_A = {optimal_allocation_A_unrestricted[0]}, x2_A = {optimal_allocation_A_unrestricted[1]}')
print(f'Maximum utility for A without restrictions: {max_utility_A_unrestricted}')

# %% [markdown]
# # Question 6a

# %% [markdown]
# We maximize the total utility in the economy, considering the utilities of both consumers A and B.

# %%
# Parameters
alpha = 1 / 3
beta = 2 / 3
omega_A1 = 0.8
omega_A2 = 0.3
omega_B1 = 1 - omega_A1
omega_B2 = 1 - omega_A2

def utility_A(x1, x2):
    return (x1 ** alpha) * (x2 ** (1 - alpha))

def utility_B(x1, x2):
    return (x1 ** beta) * (x2 ** (1 - beta))

# Objective function to maximize aggregate utility
def aggregate_utility(x):
    x1_A, x2_A = x
    x1_B = 1 - x1_A
    x2_B = 1 - x2_A
    return -(utility_A(x1_A, x2_A) + utility_B(x1_B, x2_B))

# Bounds for the variables
bounds = [(0, 1), (0, 1)]

# Initial guess
x0 = [omega_A1, omega_A2]

# Perform the optimization
result = minimize(aggregate_utility, x0, bounds=bounds)
optimal_allocation_A = result.x
max_aggregate_utility = -result.fun

print(f'Optimal allocation for A: x1_A = {optimal_allocation_A[0]}, x2_A = {optimal_allocation_A[1]}')
print(f'Maximum aggregate utility: {max_aggregate_utility}')

# Corresponding allocation for B
optimal_allocation_B = [1 - optimal_allocation_A[0], 1 - optimal_allocation_A[1]]
print(f'Optimal allocation for B: x1_B = {optimal_allocation_B[0]}, x2_B = {optimal_allocation_B[1]}')


# %% [markdown]
# # Question 6b

# %% [markdown]
# Illustrating the allocation found in previous question:

# %%
# Parameters
alpha = 1 / 3
beta = 2 / 3
omega_A1 = 0.8
omega_A2 = 0.3
omega_B1 = 1 - omega_A1
omega_B2 = 1 - omega_A2

def utility_A(x1, x2):
    return (x1 ** alpha) * (x2 ** (1 - alpha))

def utility_B(x1, x2):
    return (x1 ** beta) * (x2 ** (1 - beta))

# Function to maximize aggregate utility
def aggregate_utility(x):
    x1_A, x2_A = x
    x1_B = 1 - x1_A
    x2_B = 1 - x2_A
    return -(utility_A(x1_A, x2_A) + utility_B(x1_B, x2_B))

# Bounds for the variables
bounds = [(0, 1), (0, 1)]

# Initial guess
x0 = [omega_A1, omega_A2]

# Perform the optimization
result = minimize(aggregate_utility, x0, bounds=bounds)
x1_A_aggregate, x2_A_aggregate = result.x
x1_B_aggregate, x2_B_aggregate = 1 - x1_A_aggregate, 1 - x2_A_aggregate

print(f'Optimal allocation for A: x1_A = {x1_A_aggregate}, x2_A = {x2_A_aggregate}')
print(f'Optimal allocation for B: x1_B = {x1_B_aggregate}, x2_B = {x2_B_aggregate}')
print(f'Maximum aggregate utility: {-result.fun}')

# Plotting the optimal allocation in the Edgeworth box
plt.figure(figsize=(8, 8))
plt.scatter([x1_A_aggregate], [x2_A_aggregate], c='blue', label='Optimal allocation for A (Q6a)')
plt.scatter([x1_B_aggregate], [x2_B_aggregate], c='red', label='Optimal allocation for B (Q6a)')

# Set up the Edgeworth Box
plt.plot([0, 1], [0, 0], 'k-', lw=2)  # Bottom
plt.plot([0, 0], [0, 1], 'k-', lw=2)  # Left
plt.plot([1, 1], [0, 1], 'k-', lw=2)  # Top
plt.plot([0, 1], [1, 1], 'k-', lw=2)  # Right

# Labels and legend
plt.xlabel('$x_1^A$')
plt.ylabel('$x_2^A$')
plt.title('Optimal Allocation in the Edgeworth Box (Q6a)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# Results Recap:
# (Question 3): This is the price at which the demand from consumer B meets the supply (initially owned by consumer A), ensuring no excess demand or supply in the market for both goods.
# 
# (Question 4a): Consumer A chooses the price to maximize their utility given a set of possible prices, with consumer B responding optimally to those prices.
# 
# (Question 4b): Consumer A chooses any positive price to maximize their utility with consumer B responding optimally.
# 
# (Question 5a): Consumer A chooses the allocation to maximize their utility under the constraint that consumer B is not worse off than at their initial endowment.
# 
# (Question 5b): Consumer A chooses the allocation to maximize their utility without any restrictions.
# 
# (Question 6a): A utilitarian social planner chooses the allocation that maximizes the total utility of consumers A and B.
# 
# Discussion:
# 
# Market clearing price:
# Pros: Leads to an efficient market where all goods are sold, and all demands are satisfied.
# Cons: May not consider individual welfare or fairness. Some individuals may be worse off if the market clearing price is high or low relative to their endowment.
# 
# Maximization under set prices:
# Pros: Consumer A can optimize utility within a range of prices, representing scenarios with price control.
# Cons: The choice is limited by the set prices, and it might not lead to an optimal welfare outcome.
# 
# Maximization under any positive price:
# Pros: Ensures no party is worse off, leading to potential Pareto improvements. Fair in the sense that it respects initial endowments.
# Cons: May not reach the absolute utility maximum for A since it is constrained by B's welfare.
# 
# Pareto improvements: 
# Pros: Maximizes A's utility without restrictions, leading to the highest possible outcome for A.
# Cons: Risks leaving B worse off than at the start, which can be unfair or socially undesirable.
# 
# 
# Social Planner's Problem:
# Pros: Aims for overall social welfare maximization, considering the utilities of all individuals. Can lead to more equitable outcomes.
# Cons: Individual freedoms are not considered. It might lead to allocations that, while utility maximizing on the whole, leave one party significantly worse off than they would prefer.
# 
# 

# %% [markdown]
# # Question 7

# %% [markdown]
# np.random.uniform(0, 1, N) generates N random numbers from a uniform distribution between 0 and 1.
# We then use plt.scatter to plot these pairs in a scatter plot, where each point represents an element of the set WW.
# The labels and title are added for clarity.

# %%
import numpy as np
import matplotlib.pyplot as plt

# Number of elements in the set
N = 50

# Generate N random values for ω1^A and ω2^A, each uniformly distributed between 0 and 1
omega_1A = np.random.uniform(0, 1, N)
omega_2A = np.random.uniform(0, 1, N)

# Plot these pairs
plt.figure(figsize=(8, 8))
plt.scatter(omega_1A, omega_2A, c='blue', label='Random Set W')
plt.xlabel('$\omega_1^A$')
plt.ylabel('$\omega_2^A$')
plt.title('Random Set W with 50 Elements')
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# # Question 8

# %% [markdown]
# In this code:
# 
# We generate a set WW with 50 elements, each representing a different initial endowment for consumer A.
# We then plot these endowments in an Edgeworth box for both consumer A (in blue) and consumer B (in red), assuming the economy only consists of these two consumers and that all resources not owned by A are owned by B.
# This visualizes the different potential market equilibrium allocations assuming that each initial endowment leads to an equilibrium where the goods are not traded (this is a simplification and may not hold if there are specific market mechanisms or preferences at play).

# %%
# Number of elements
N = 50

# Random endowments
omega_1A = np.random.uniform(0, 1, N)
omega_2A = np.random.uniform(0, 1, N)

# Parameters
alpha = 1 / 3
beta = 2 / 3
N = 50

def utility_A(x1, x2):
    return (x1 ** alpha) * (x2 ** (1 - alpha))

def utility_B(x1, x2):
    return (x1 ** beta) * (x2 ** (1 - beta))

# Demand functions
def demand_A(p1, p2, omega_A1, omega_A2, alpha):
    x1_A = alpha * (p1 * omega_A1 + p2 * omega_A2) / p1
    x2_A = (1 - alpha) * (p1 * omega_A1 + p2 * omega_A2) / p2
    return x1_A, x2_A

def demand_B(p1, p2, omega_B1, omega_B2, beta):
    x1_B = beta * (p1 * omega_B1 + p2 * omega_B2) / p1
    x2_B = (1 - beta) * (p1 * omega_B1 + p2 * omega_B2) / p2
    return x1_B, x2_B

# Function to calculate total market clearing error
def total_error(p1, omega_A1, omega_A2, omega_B1, omega_B2):
    x1_A, x2_A = demand_A(p1, 1, omega_A1, omega_A2, alpha)
    x1_B, x2_B = demand_B(p1, 1, omega_B1, omega_B2, beta)
    epsilon1 = x1_A + x1_B - 1
    epsilon2 = x2_A + x2_B - 1
    return abs(epsilon1) + abs(epsilon2)

# Generate random endowments
random_endowments = np.random.uniform(0, 1, (N, 2))
market_equilibria = []

# Find market equilibrium for each set of endowments
for omega_A in random_endowments:
    omega_A1, omega_A2 = omega_A
    omega_B1 = 1 - omega_A1
    omega_B2 = 1 - omega_A2
    
    result = minimize(total_error, 1, args=(omega_A1, omega_A2, omega_B1, omega_B2), bounds=[(0.01, 10)])
    p1_star = result.x[0]
    
    x1_A_star, x2_A_star = demand_A(p1_star, 1, omega_A1, omega_A2, alpha)
    market_equilibria.append((x1_A_star, x2_A_star))

market_equilibria = np.array(market_equilibria)

# Plotting the market equilibrium allocations in the Edgeworth box
plt.figure(figsize=(8, 8))
plt.scatter(market_equilibria[:, 0], market_equilibria[:, 1], c='blue', label='Equilibrium allocations for A')
plt.scatter(1 - market_equilibria[:, 0], 1 - market_equilibria[:, 1], c='red', label='Equilibrium allocations for B', alpha=0.6)

# Set up the Edgeworth Box
plt.plot([0, 1], [0, 0], 'k-', lw=2)  # Bottom
plt.plot([0, 0], [0, 1], 'k-', lw=2)  # Left
plt.plot([1, 1], [0, 1], 'k-', lw=2)  # Top
plt.plot([0, 1], [1, 1], 'k-', lw=2)  # Right

# Labels and legend
plt.xlabel('$x_1^A$')
plt.ylabel('$x_2^A$')
plt.title('Market Equilibrium Allocations in the Edgeworth Box')
plt.legend()
plt.grid(True)
plt.show()






