# %% [markdown]
# **Table of contents**<a id='toc0_'></a>    
# - 1. [Problem 1: Production economy and CO2 taxation](#toc1_)    
# - 2. [Problem 2: Career choice model](#toc2_)    
# - 3. [Problem 3: Barycentric interpolation](#toc3_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=true
# 	anchor=true
# 	flat=false
# 	minLevel=2
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %%
# Write your code here
import numpy as np
from types import SimpleNamespace

# %% [markdown]
# ## 1. <a id='toc1_'></a>[Problem 1: Production economy and CO2 taxation](#toc0_)

# %% [markdown]
# Consider a production economy with two firms indexed by $j \in \{1,2\}$. Each produce its own good. They solve
# 
# $$
# \begin{align*}
# \max_{y_{j}}\pi_{j}&=p_{j}y_{j}-w_{j}\ell_{j}\\\text{s.t.}\;&y_{j}=A\ell_{j}^{\gamma}.
# \end{align*}
# $$
# 
# Optimal firm behavior is
# 
# $$
# \begin{align*}
# \ell_{j}^{\star}(w,p_{j})&=\left(\frac{p_{j}A\gamma}{w}\right)^{\frac{1}{1-\gamma}} \\
# y_{j}^{\star}(w,p_{j})&=A\left(\ell_{j}^{\star}(w,p_{j})\right)^{\gamma}
# \end{align*}
# $$
# 
# The implied profits are
# 
# $$
# \pi_{j}^*(w,p_{j})=\frac{1-\gamma}{\gamma}w\cdot\left(\frac{p_{j}A\gamma}{w}\right)^{\frac{1}{1-\gamma}}
# $$
# 
# A single consumer supplies labor, and consumes the goods the firms produce. She also recieves the implied profits of the firm.<br>
# She solves:
# 
# $$
# \begin{align*}
# U(p_1,p_2,w,\tau,T) = \max_{c_{1},c_{2},\ell} & \log(c_{1}^{\alpha}c_{2}^{1-\alpha})-\nu\frac{\ell^{1+\epsilon}}{1+\epsilon} \\
# \text{s.t.}\,\,\,&p_{1}c_{1}+(p_{2}+\tau)c_{2}=w\ell+T+\pi_{1}^*(w,p_{1})+\pi_{2}^*(w,p_{2})
# \end{align*}
# $$
# 
# where $\tau$ is a tax and $T$ is lump-sum transfer. <br>
# For a given $\ell$, it can be shown that optimal behavior is
# 
# $$
# \begin{align*}
# c_{1}(\ell)&=\alpha\frac{w\ell+T+\pi_{1}^*(w,p_{1})+\pi_{2}^*(w,p_{2})}{p_{1}} \\
# c_{2}(\ell)&=(1-\alpha)\frac{w\ell+T+\pi_{1}^*(w,p_{1})+\pi_{2}^*(w,p_{2})}{p_{2}+\tau} \\
# \end{align*}
# $$
# Such that optimal behavior is:
# $$
# \ell^* = \underset{\ell}{\arg\max} \log(\left(c_{1}(\ell)\right)^{\alpha}\cdot \left(c_{2}(\ell)\right)^{1-\alpha})-\nu\frac{\ell^{1+\epsilon}}{1+\epsilon} 
# $$
# With optimal consumption:
# $$
# \begin{align*}
# c_1^*=c_{1}(\ell^*) \\
# c_2^*=c_{2}(\ell^*)\\
# \end{align*}
# $$
# 
# 
# The government chooses $\tau$ and balances its budget so $T=\tau c_2^*$. We initially set $\tau,T=0$.
# 
# Market clearing requires:
# 
# 1. Labor market: $\ell^* = \ell_1^* + \ell_2^*$
# 1. Good market 1: $c_1^* = y_1^*$
# 1. Good market 2: $c_2^* = y_2^*$
# 

# %% [markdown]
# **Question 1:** Check market clearing conditions for $p_1$ in `linspace(0.1,2.0,10)` and $p_2$ in `linspace(0.1,2.0,10)`. We choose $w=1$ as numeraire.

# %%
par = SimpleNamespace()

# firms
par.A = 1.0
par.gamma = 0.5

# households
par.alpha = 0.3
par.nu = 1.0
par.epsilon = 2.0

# government
par.tau = 0.0
par.T = 0.0

# Question 3
par.kappa = 0.1

# %%
import numpy as np
from types import SimpleNamespace

# First we define the parameters
par = SimpleNamespace()
par.A = 1.0
par.gamma = 0.5
par.alpha = 0.3
par.nu = 1.0
par.epsilon = 2.0
par.tau = 0.0
par.T = 0.0
par.kappa = 0.1
w = 1  # numeraire

# Then we define functions for optimal firm behavior and profits
def optimal_labor(w, p, A, gamma):
    return (p * A * gamma / w)**(1 / (1 - gamma))

def optimal_output(w, p, A, gamma):
    return A * optimal_labor(w, p, A, gamma)**gamma

def profits(w, p, A, gamma):
    return w * (p * A * gamma / w)**(1 / (1 - gamma)) * (1 - gamma)

# Then we define functions for consumer optimization
def consumer_utility(p1, p2, w, tau, T, alpha, nu, epsilon):
    def utility(c1, c2, ell):
        return np.log(c1**alpha * c2**(1 - alpha)) - nu * ell**(1 + epsilon) / (1 + epsilon)
    
    ell_star = (w + T + profits(w, p1, par.A, par.gamma) + profits(w, p2, par.A, par.gamma)) / (p1 * alpha + (p2 + tau) * (1 - alpha))
    c1_star = alpha * (w * ell_star + T + profits(w, p1, par.A, par.gamma) + profits(w, p2, par.A, par.gamma)) / p1
    c2_star = (1 - alpha) * (w * ell_star + T + profits(w, p1, par.A, par.gamma) + profits(w, p2, par.A, par.gamma)) / (p2 + tau)
    
    return utility(c1_star, c2_star, ell_star), c1_star, c2_star, ell_star

# Now we check market clearing conditions for a range of prices
# Following generates 10 equally spaced values between 0.1 and 2.0 for both p1 and p2
p1_values = np.linspace(0.1, 2.0, 10)
p2_values = np.linspace(0.1, 2.0, 10)
results = [] # Create an empty list 

# Then we Loop through different price combinations and make sure that the market clearing requirements are uphold
for p1 in p1_values:
    for p2 in p2_values:
        U, c1, c2, ell = consumer_utility(p1, p2, w, par.tau, par.T, par.alpha, par.nu, par.epsilon)
        y1 = optimal_output(w, p1, par.A, par.gamma)
        y2 = optimal_output(w, p2, par.A, par.gamma)
        labor_market_clearing = np.isclose(ell, optimal_labor(w, p1, par.A, par.gamma) + optimal_labor(w, p2, par.A, par.gamma), rtol=1e-4)
        good1_market_clearing = np.isclose(c1, y1, rtol=1e-4)
        good2_market_clearing = np.isclose(c2, y2, rtol=1e-4)
        
        results.append((p1, p2, labor_market_clearing, good1_market_clearing, good2_market_clearing))

# Displaying results for Question 1
for result in results:
    print(f"p1: {result[0]:.2f}, p2: {result[1]:.2f}, Labor Market Clearing: {result[2]}, Good 1 Market Clearing: {result[3]}, Good 2 Market Clearing: {result[4]}")
   

# %% [markdown]
# This indicates that with the given parameters and price ranges, the market for neither Good 1 nor Good 2 reaches equilibrium. Therefore, we can conclude that under the current conditions and assumptions, there are no price combinations within the specified ranges that lead to both goods markets clearing simultaneously.

# %% [markdown]
# **Question 2:** Find the equilibrium prices $p_1$ and $p_2$.<br>
# *Hint: you can use Walras' law to only check 2 of the market clearings*

# %%
from scipy.optimize import fsolve

# First we define the market clearing functions
def market_clearing(prices):
    p1, p2 = prices
    _, c1, c2, ell = consumer_utility(p1, p2, w, par.tau, par.T, par.alpha, par.nu, par.epsilon)
    y1 = optimal_output(w, p1, par.A, par.gamma)
    y2 = optimal_output(w, p2, par.A, par.gamma)
    
    # Then we define the two market clearing conditions
    labor_market_clearing = ell - (optimal_labor(w, p1, par.A, par.gamma) + optimal_labor(w, p2, par.A, par.gamma))
    good1_market_clearing = c1 - y1
    
    return [labor_market_clearing, good1_market_clearing]

# Find equilibrium prices using fsolve
initial_guess = [1.0, 1.0]
equilibrium_prices = fsolve(market_clearing, initial_guess)

print(f"Equilibrium prices: p1 = {equilibrium_prices[0]}, p2 = {equilibrium_prices[1]}")

# %% [markdown]
# The equilibrium prices were determined by ensuring the market clearing conditions for both the labor market and the goods market for good 1. Using Walras law we know that if good 1 is in market equilibrium then good is aswell.
# 
# These prices ensure that the total labor supplied by the consumer matches the total labor demanded by the firms, and that the consumer's demand for good 1 equals its supply. The equilibrium prices provide a balance between the production decisions of the firms and the consumption choices of the consumer in the economy.

# %% [markdown]
# Assume the government care about the social welfare function:
# 
# $$
# SWF = U - \kappa y_2^*
# $$

# %% [markdown]
# Here $\kappa$ measures the social cost of carbon emitted by the production of $y_2$ in equilibrium.

# %% [markdown]
# **Question 3:** What values of $\tau$ and (implied) $T$ should the government choose to maximize $SWF$?

# %%
from scipy.optimize import minimize

# Functions for consumer optimization
def consumer_utility(p1, p2, w, tau, T, alpha, nu, epsilon):
    def utility(c1, c2, ell):
        return np.log(c1**alpha * c2**(1 - alpha)) - nu * ell**(1 + epsilon) / (1 + epsilon)
    
    ell_star = (w + T + profits(w, p1, par.A, par.gamma) + profits(w, p2, par.A, par.gamma)) / (p1 * alpha + (p2 + tau) * (1 - alpha))
    c1_star = alpha * (w * ell_star + T + profits(w, p1, par.A, par.gamma) + profits(w, p2, par.A, par.gamma)) / p1
    c2_star = (1 - alpha) * (w * ell_star + T + profits(w, p1, par.A, par.gamma) + profits(w, p2, par.A, par.gamma)) / (p2 + tau)
    
    return utility(c1_star, c2_star, ell_star)

# Social welfare function to be maximized
def social_welfare(vars, par):
    tau, T = vars
    par.tau = tau
    par.T = T
    p1, p2 = 1.0, 1.0 
    U = consumer_utility(p1, p2, w, tau, T, par.alpha, par.nu, par.epsilon)
    y2 = optimal_output(w, p2, par.A, par.gamma)
    SWF = U - par.kappa * y2
    return -SWF  # Minimize the negative of SWF for maximization

# Initial guess for tau and T
initial_guess = [0.0, 0.0]

# Bounds for tau and T (assuming non-negative values)
bounds = [(0, None), (0, None)]

# Find the optimal tau and T
result = minimize(social_welfare, initial_guess, args=(par), bounds=bounds)

optimal_tau, optimal_T = result.x

# Calculate the utility with the optimal tau and T
par.tau = optimal_tau
par.T = optimal_T
p1, p2 = 1.0, 1.0  
U = consumer_utility(p1, p2, w, par.tau, par.T, par.alpha, par.nu, par.epsilon)

# Display the optimal values and utility
print(f"Optimal tax (tau): {optimal_tau}")
print(f"Optimal transfer (T): {optimal_T}")
print(f"Consumer utility (U): {U}")


# %% [markdown]
# The optimization results suggest that the government should impose a small tax on good 2 while providing no lump-sum transfer to maximize social welfare. The negative utility, reflects the consumer's response to the optimized tax policy, balancing between consumption and labor supply within the given economic model. This approach helps in understanding the trade-offs and impacts of tax policies on overall welfare in the production economy.

# %% [markdown]
# ## 2. <a id='toc2_'></a>[Problem 2: Career choice model](#toc0_)

# %% [markdown]
# Consider a graduate $i$ making a choice between entering $J$ different career tracks. <br>
# Entering career $j$ yields utility $u^k_{ij}$. This value is unknown to the graduate ex ante, but will ex post be: <br>
# $$
#     u_{i,j}^k = v_{j} + \epsilon_{i,j}^k
# $$
# 
# They know that $\epsilon^k_{i,j}\sim \mathcal{N}(0,\sigma^2)$, but they do not observe $\epsilon^k_{i,j}$ before making their career choice. <br>

# %% [markdown]
# Consider the concrete case of $J=3$ with:
# $$
# \begin{align*}
#     v_{1} &= 1 \\
#     v_{2} &= 2 \\
#     v_{3} &= 3
# \end{align*}
# $$

# %% [markdown]
# If the graduates know the values of $v_j$ and the distribution of $\epsilon_{i,j}^k$, they can calculate the expected utility of each career track using simulation: <br>
# $$
#     \mathbb{E}\left[ u^k_{i,j}\vert v_j \right] \approx v_j + \frac{1}{K}\sum_{k=1}^K \epsilon_{i,j}^k
# $$

# %%
import numpy as np
import pandas as pd
from types import SimpleNamespace

par = SimpleNamespace()
par.J = 3
par.N = 10
par.K = 10000

par.F = np.arange(1,par.N+1)
par.sigma = 2

par.v = np.array([1,2,3])
par.c = 1

# %% [markdown]
# **Question 1:** Simulate and calculate expected utility and the average realised utility for $K=10000$ draws, for each career choice $j$.
# 

# %% [markdown]
# We are looking into the dynamics of a career choice model, where we are asked to simulate and calculate expected and realized utility, where the graduates know $v_j$ and the distribution of $\epsilon_j$. This is done below.

# %%
# Simulate and calculate expected utility and average realized utility
def simulate_utilities(par):
    # Simulate epsilons
    epsilon = np.random.normal(0, par.sigma, (par.K, par.J))

    # Calculate expected utility for each career track j
    expected_utility = par.v + (1 / par.K) * np.sum(epsilon, axis=0)

    # Calculate the average realized utility
    average_realized_utility = par.v + np.mean(epsilon, axis=0)

    return expected_utility, average_realized_utility

expected_utility, average_realized_utility = simulate_utilities(par)

# Display the results
df = pd.DataFrame({
    'Career Track': np.arange(1, par.J + 1),
    'Expected Utility': expected_utility,
    'Average Realized Utility': average_realized_utility
})

print(df)

# %% [markdown]
# We get utility values very close to the true value, as we run the simulation 10.000 times. This is as the graduates know the distribution of the error term, where the expected value is zero, hence we are left with utility very close to $v_j$.

# %% [markdown]
# Now consider a new scenario: Imagine that the graduate does not know $v_j$. The *only* prior information they have on the value of each job, comes from their $F_{i}$ friends that work in each career $j$. After talking with them, they know the average utility of their friends (which includes their friends' noise term), giving them the prior expecation: <br>
# $$
# \tilde{u}^k_{i,j}\left( F_{i}\right) = \frac{1}{F_{i}}\sum_{f=1}^{F_{i}} \left(v_{j} + \epsilon^k_{f,j}\right), \; \epsilon^k_{f,j}\sim \mathcal{N}(0,\sigma^2)
# $$
# For ease of notation consider that each graduate have $F_{i}=i$ friends in each career. <br>

# %% [markdown]
# For $K$ times do the following: <br>
# 1. For each person $i$ draw $J\cdot F_i$ values of $\epsilon_{f,j}^{k}$, and calculate the prior expected utility of each career track, $\tilde{u}^k_{i,j}\left( F_{i}\right)$. <br>
# Also draw their own $J$ noise terms, $\epsilon_{i,j}^k$
# 1. Each person $i$ chooses the career track with the highest expected utility: $$j_i^{k*}= \arg\max_{j\in{1,2\dots,J}}\left\{ \tilde{u}^k_{i,j}\left( F_{i}\right)\right\} $$
# 1. Store the chosen careers: $j_i^{k*}$, the prior expectation of the value of their chosen career: $\tilde{u}^k_{i,j=j_i^{k*}}\left( F_{i}\right)$, and the realized value of their chosen career track: $u^k_{i,j=j_i^{k*}}=v_{j=j_i^{k*}}+\epsilon_{i,j=j_i^{k*}}^k$.

# %% [markdown]
# Chosen values will be: <br>
# $i\in\left\{1,2\dots,N\right\}, N=10$ <br>
# $F_i = i$<br>
# So there are 10 graduates. The first has 1 friend in each career, the second has 2 friends, ... the tenth has 10 friends.

# %% [markdown]
# **Question 2:** Simulate and visualize: For each type of graduate, $i$, the share of graduates choosing each career, the average subjective expected utility of the graduates, and the average ex post realized utility given their choice. <br>
# That is, calculate and visualize: <br>
# $$
# \begin{align*}
#     \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}\left\{ j=j_i^{k*} \right\}  \;\forall j\in\left\{1,2,\dots,J\right\}
# \end{align*}
# $$
# $$
# \begin{align*}
#     \frac{1}{K} \sum_{k=1}^{K} \tilde{u}^k_{ij=j_i^{k*}}\left( F_{i}\right)
# \end{align*}
# $$
# And 
# $$
# \begin{align*}
#     \frac{1}{K} \sum_{k=1}^{K} u^k_{ij=j_i^{k*}} 
# \end{align*}
# $$
# For each graduate $i$.

# %% [markdown]
# We now consider the scenario where $v_j$ is not known to the graudates. Start by calculating and storing the chosen career paths. 

# %%
def simulate_new_scenario(par):
    results = []

    for k in range(par.K):
        for i in range(1, par.N + 1):
            # Draw J * F_i values of epsilon
            epsilon_friends = np.random.normal(0, par.sigma, (par.F[i - 1], par.J))
            epsilon_own = np.random.normal(0, par.sigma, par.J)

            # Calculate the prior expected utility of each career track
            prior_expected_utility = par.v + np.mean(epsilon_friends, axis=0)

            # Choose the career track with the highest expected utility
            chosen_career = np.argmax(prior_expected_utility)

            # Store the chosen career, prior expectation, and realized value
            realized_value = par.v[chosen_career] + epsilon_own[chosen_career]
            results.append({
                'Graduate': i,
                'Chosen Career': chosen_career + 1,
                'Prior Expected Utility': prior_expected_utility[chosen_career],
                'Realized Utility': realized_value
            })

    return pd.DataFrame(results)

# %% [markdown]
# Which we print, where we see prior expected and realized utility get closer as the graduates obtain more friends, which makes sense in the model.

# %%
# Simulate the new scenario
df_results = simulate_new_scenario(par)

# Display the results
df_summary = df_results.groupby('Graduate').agg({
    'Prior Expected Utility': 'mean',
    'Realized Utility': 'mean'
}).reset_index()

print(df_summary)


# %% [markdown]
# Next, we simulate and visualize the development and impact of obtaining more friends.

# %%
import matplotlib.pyplot as plt

# Initialize dictionaries to store results
choice_shares = {j: np.zeros(par.N) for j in range(1, par.J + 1)}
avg_expected_utility = np.zeros(par.N)
avg_realized_utility = np.zeros(par.N)

# Simulate the new scenario
for k in range(par.K):
    for i in range(1, par.N + 1):
        # Draw J * F_i values of epsilon
        epsilon_friends = np.random.normal(0, par.sigma, (par.F[i - 1], par.J))
        epsilon_own = np.random.normal(0, par.sigma, par.J)

        # Calculate the prior expected utility of each career track
        prior_expected_utility = par.v + np.mean(epsilon_friends, axis=0)

        # Choose the career track with the highest expected utility
        chosen_career = np.argmax(prior_expected_utility)

        # Increment choice share for the chosen career
        choice_shares[chosen_career + 1][i - 1] += 1

        # Store the prior expectation and realized value for averages
        avg_expected_utility[i - 1] += prior_expected_utility[chosen_career]
        realized_value = par.v[chosen_career] + epsilon_own[chosen_career]
        avg_realized_utility[i - 1] += realized_value

# Calculate the averages
for j in choice_shares:
    choice_shares[j] /= par.K

avg_expected_utility /= par.K
avg_realized_utility /= par.K

# Convert choice shares to a DataFrame for easier plotting
df_choice_shares = pd.DataFrame(choice_shares)
df_choice_shares['Graduate'] = np.arange(1, par.N + 1)
df_choice_shares = df_choice_shares.melt(id_vars='Graduate', var_name='Career Track', value_name='Share')

# Plot the choice shares
plt.figure(figsize=(10, 6))
for career in range(1, par.J + 1):
    plt.plot(df_choice_shares[df_choice_shares['Career Track'] == career]['Graduate'],
             df_choice_shares[df_choice_shares['Career Track'] == career]['Share'], label=f'Career {career}')
plt.xlabel('Friends')
plt.ylabel('Share of Graduates Choosing Each Career')
plt.title('Share of Graduates Choosing Each Career by # of Friends')
plt.legend()
plt.show()

# Plot the average  expected utility and realized utility
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, par.N + 1), avg_expected_utility, label='Average Expected Utility')
plt.plot(np.arange(1, par.N + 1), avg_realized_utility, label='Average Realized Utility')
plt.xlabel('Friends')
plt.ylabel('Utility')
plt.title('Average Expected Utility and Realized Utility by # of Friends')
plt.legend()
plt.show()


# %% [markdown]
# First, as a graudate obtain more friends, the probability (share) of choosing career path 3 increases. Better knowledge from more friends makes the graduate decide more wisely. Secondly, we see that expected utility and realized utility behave inversly. So, the fewer friends (the lower # graduate), the higher one expects its utility to be, but actually end up significantly lower. As friends increase, the gap narrows between expected and realized utility, though level of expected utility still maintains a higher relative level. We recall this is from 10.000 simulations, and single simulations can deviate greatly.

# %% [markdown]
# After a year of working in their career, the graduates learn $u^k_{ij}$ for their chosen job $j_i^{k*}$ perfectly. <br>
# The can switch to one of the two remaining careers, for which they have the same prior as before, but it will now include a switching cost of $c$ which is known.
# Their new priors can be written as: 
# $$
# \tilde{u}^{k,2}_{ij}\left( F_{i}\right) = \begin{cases}
#             \tilde{u}^k_{ij}\left( F_{i}\right)-c & \text{if } j \neq j_i^{k*} \\
#             u_{ij=j_i^{k*}} & \text{if } j = j_i^{k*}
#         \end{cases}
# $$

# %% [markdown]
# We will set $c=1$.

# %% [markdown]
# Their realized utility will be: <br>
# $$
# u^{k,2}_{ij}= \begin{cases}
#             u_{ij}^k -c & \text{if } j \neq j_i^{k*} \\
#             u_{ij=j_i^{k*}} & \text{if } j = j_i^{k*}
#         \end{cases}
# $$

# %% [markdown]
# **Question 3:** Following the same approach as in question 2, find the new optimal career choice for each $i$, $k$. Then for each $i$, calculate the average subjective expected utility from their new optimal career choice, and the ex post realized utility of that career. Also, for each $i$, calculate the share of graduates that chooses to switch careers, conditional on which career they chose in the first year. <br>

# %% [markdown]
# For this last part of Problem 2, we wish to investigate what happens, if the graduates can switch careers after 1 year in another career, thus knowing the utility of that specific career, where there's a switching cost of c = 1. So we simulate the model again, where we also include the switching cost. 

# %%
# Initial simulation to get career choices and utilities after the first year
def simulate_initial_choices(par):
    np.random.seed(2024)
    career_choices = np.zeros((par.K, par.N), dtype=int)
    prior_utility = np.zeros((par.K, par.N))
    realized_utility = np.zeros((par.K, par.N))

    for k in range(par.K):
        for i in range(par.N):
            Fi = i + 1
            eps = np.random.normal(0, par.sigma, (Fi, par.J))
            prior_exp_utility = np.mean(par.v + eps, axis=0)
            own_eps = np.random.normal(0, par.sigma, par.J)
            own_utility = par.v + own_eps
            chosen_career = np.argmax(prior_exp_utility)
            career_choices[k, i] = chosen_career
            prior_utility[k, i] = prior_exp_utility[chosen_career]
            realized_utility[k, i] = own_utility[chosen_career]

    return career_choices, prior_utility, realized_utility

# Updating simulation to account for switching after the first year
def simulate_switching_choices(par, career_choices, realized_utility):
    np.random.seed(2024)
    new_career_choices = np.zeros_like(career_choices)
    new_prior_utility = np.zeros_like(realized_utility)
    new_realized_utility = np.zeros_like(realized_utility)
    switching_cost = par.c

    for k in range(par.K):
        for i in range(par.N):
            current_career = career_choices[k, i]
            Fi = i + 1
            eps = np.random.normal(0, par.sigma, (Fi, par.J))
            prior_exp_utility = np.mean(par.v + eps, axis=0)
            own_eps = np.random.normal(0, par.sigma, par.J)
            own_utility = par.v + own_eps

            new_prior_exp_utility = prior_exp_utility - switching_cost
            new_prior_exp_utility[current_career] = realized_utility[k, i]

            new_chosen_career = np.argmax(new_prior_exp_utility)
            new_career_choices[k, i] = new_chosen_career
            new_prior_utility[k, i] = new_prior_exp_utility[new_chosen_career]
            new_realized_utility[k, i] = own_utility[new_chosen_career]

    return new_career_choices, new_prior_utility, new_realized_utility

# Calculating the share of graduates switching careers for each initial career choice
def calculate_switching_shares(career_choices, new_career_choices, par):
    switch_counts = np.zeros((par.J, par.N))
    total_counts = np.zeros((par.J, par.N))

    for j in range(par.J):
        for i in range(par.N):
            initial_career_indices = career_choices[:, i] == j
            switch_counts[j, i] = np.sum(career_choices[initial_career_indices, i] != new_career_choices[initial_career_indices, i])
            total_counts[j, i] = np.sum(initial_career_indices)

    switch_shares = switch_counts / total_counts
    return switch_shares

# Calculating the average expected utility and average realized utility for each type of graduate
def calculate_average_utilities(new_prior_utility, new_realized_utility):
    avg_new_prior_utility = np.mean(new_prior_utility, axis=0)
    avg_new_realized_utility = np.mean(new_realized_utility, axis=0)
    return avg_new_prior_utility, avg_new_realized_utility

# Run the simulation
career_choices, prior_utility, realized_utility = simulate_initial_choices(par)
new_career_choices, new_prior_utility, new_realized_utility = simulate_switching_choices(par, career_choices, realized_utility)
switch_shares = calculate_switching_shares(career_choices, new_career_choices, par)
avg_new_prior_utility, avg_new_realized_utility = calculate_average_utilities(new_prior_utility, new_realized_utility)

# %% [markdown]
# Now we can calcuate and plot expected and realized utility, which give that both expected and realized utility increase with number of friends, when the graduates can switch careers. And contrary to before (in Question 2), the level of realized utility is higher than expected utility for all graduates.

# %%
plt.figure(figsize=(10, 6))
print("Average Subjective Expected Utility by # of Friends:")
print(avg_new_prior_utility)

print("\nAverage Realized Utility by # of Friends:")
print(avg_new_realized_utility)

plt.plot(par.F, avg_new_prior_utility, label='Average Expected Utility')
plt.plot(par.F, avg_new_realized_utility, label='Average Realized Utility')

plt.xlabel('Number of Friends (Fi)')
plt.ylabel('Utility')
plt.title('Average Expected Utility and Realized Utility by # of Friends')
plt.legend()
plt.show()

# %% [markdown]
# And then I want to calculate and plot three graphs showing share of graduates swiching careers, based on initial career choice.

# %%
# Printing the results.
print("\nShare of Graduates Switching Careers by Initial Career Choice and # of Friends:")
for j in range(par.J):
    print(f"Career {j + 1}: {switch_shares[j, :]}")

# Plotting the results
plt.figure(figsize=(10, 6))
labels = ['Initial Career 1', 'Initial Career 2', 'Initial Career 3']

for j in range(par.J):
    plt.plot(par.F, switch_shares[j, :], label=labels[j])

plt.xlabel('Number of Friends')
plt.ylabel('Share of Graduates Switching Careers')
plt.title('Share of Graduates Switching Careers by Initial Career Choice')
plt.legend()
plt.show()

# %% [markdown]
# The graph shows what we would expect, that the higher initial career choice (and thus higher utility) translates into a smaller share of those graduates switching career after the first year, which is common for all graduates.

# %% [markdown]
# ## 3. <a id='toc3_'></a>[Problem 3: Barycentric interpolation](#toc0_)

# %% [markdown]
# **Problem:** We have a set of random points in the unit square,
# 
# $$
# \mathcal{X} = \{(x_1,x_2)\,|\,x_1\sim\mathcal{U}(0,1),x_2\sim\mathcal{U}(0,1)\}.
# $$
# 
# For these points, we know the value of some function $f(x_1,x_2)$,
# 
# $$
# \mathcal{F} = \{f(x_1,x_2) \,|\, (x_1,x_2) \in \mathcal{X}\}.
# $$
# 
# Now we want to approximate the value $f(y_1,y_2)$ for some  $y=(y_1,y_2)$, where $y_1\sim\mathcal{U}(0,1)$ and $y_2\sim\mathcal{U}(0,1)$.
# 
# **Building block I**
# 
# For an arbitrary triangle $ABC$ and a point $y$, define the so-called barycentric coordinates as:
# 
# $$
# \begin{align*}
#   r^{ABC}_1 &= \frac{(B_2-C_2)(y_1-C_1) + (C_1-B_1)(y_2-C_2)}{(B_2-C_2)(A_1-C_1) + (C_1-B_1)(A_2-C_2)} \\
#   r^{ABC}_2 &= \frac{(C_2-A_2)(y_1-C_1) + (A_1-C_1)(y_2-C_2)}{(B_2-C_2)(A_1-C_1) + (C_1-B_1)(A_2-C_2)} \\
#   r^{ABC}_3 &= 1 - r_1 - r_2.
# \end{align*}
# $$
# 
# If $r^{ABC}_1 \in [0,1]$, $r^{ABC}_2 \in [0,1]$, and $r^{ABC}_3 \in [0,1]$, then the point is inside the triangle.
# 
# We always have $y = r^{ABC}_1 A + r^{ABC}_2 B + r^{ABC}_3 C$.
# 
# **Building block II**
# 
# Define the following points:
# 
# $$
# \begin{align*}
# A&=\arg\min_{(x_{1},x_{2})\in\mathcal{X}}\sqrt{\left(x_{1}-y_{1}\right)^{2}+\left(x_{2}-y_{2}\right)^{2}}\text{ s.t. }x_{1}>y_{1}\text{ and }x_{2}>y_{2}\\
# B&=\arg\min_{(x_{1},x_{2})\in\mathcal{X}}\sqrt{\left(x_{1}-y_{1}\right)^{2}+\left(x_{2}-y_{2}\right)^{2}}\text{ s.t. }x_{1}>y_{1}\text{ and }x_{2}<y_{2}\\
# C&=\arg\min_{(x_{1},x_{2})\in\mathcal{X}}\sqrt{\left(x_{1}-y_{1}\right)^{2}+\left(x_{2}-y_{2}\right)^{2}}\text{ s.t. }x_{1}<y_{1}\text{ and }x_{2}<y_{2}\\
# D&=\arg\min_{(x_{1},x_{2})\in\mathcal{X}}\sqrt{\left(x_{1}-y_{1}\right)^{2}+\left(x_{2}-y_{2}\right)^{2}}\text{ s.t. }x_{1}<y_{1}\text{ and }x_{2}>y_{2}.
# \end{align*}
# $$
# 
# **Algorithm:**
# 
# 1. Compute $A$, $B$, $C$, and $D$. If not possible return `NaN`.
# 1. If $y$ is inside the triangle $ABC$ return $r^{ABC}_1 f(A) + r^{ABC}_2 f(B) + r^{ABC}_3 f(C)$.
# 1. If $y$ is inside the triangle $CDA$ return $r^{CDA}_1 f(C) + r^{CDA}_2 f(D) + r^{CDA}_3 f(A)$.
# 1. Return `NaN`.
# 
# 

# %% [markdown]
# **Sample:**

# %%
rng = np.random.default_rng(2024)

X = rng.uniform(size=(50,2))
y = rng.uniform(size=(2,))


# %% [markdown]
# **Questions 1:** Find $A$, $B$, $C$ and $D$. Illustrate these together with $X$, $y$ and the triangles $ABC$ and $CDA$.

# %%
import numpy as np
import matplotlib.pyplot as plt

# Function to compute barycentric coordinates
def barycentric_coords(A, B, C, P):
    denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    r1 = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denom
    r2 = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denom
    r3 = 1 - r1 - r2
    return r1, r2, r3
# The reason we use 0 instead of 1 in B[] etc. is that in python 0 = 1 and 1 = 2 etc.

# Sample data
rng = np.random.default_rng(2024)
X = rng.uniform(size=(50, 2))
y = rng.uniform(size=(2,))

# Defining points A, B, C, D
A = min([x for x in X if x[0] > y[0] and x[1] > y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
B = min([x for x in X if x[0] > y[0] and x[1] < y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
C = min([x for x in X if x[0] < y[0] and x[1] < y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
D = min([x for x in X if x[0] < y[0] and x[1] > y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))

# Compute barycentric coordinates for point y with respect to triangles ABC and CDA
rABC1, rABC2, rABC3 = barycentric_coords(A, B, C, y)
rCDA1, rCDA2, rCDA3 = barycentric_coords(C, D, A, y)

# Plotting
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Random Points')
plt.scatter(*y, color='red', label='Point y', zorder=5)
plt.scatter(*A, color='green', label='Point A', zorder=5)
plt.scatter(*B, color='green', label='Point B', zorder=5)
plt.scatter(*C, color='green', label='Point C', zorder=5)
plt.scatter(*D, color='green', label='Point D', zorder=5)

plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color='black', label='Triangle ABC')
plt.plot([C[0], D[0], A[0], C[0]], [C[1], D[1], A[1], C[1]], color='purple', label='Triangle CDA')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Barycentric Interpolation Visualization')
plt.grid(True)
plt.show()

# Print results
print("Point y:", y)
print("Triangle ABC: A =", A, ", B =", B, ", C =", C)
print("Triangle CDA: C =", C, ", D =", D, ", A =", A)

# %% [markdown]
# Note that random point are equal to point X. 

# %% [markdown]
# **Question 2:** Compute the barycentric coordinates of the point $y$ with respect to the triangles $ABC$ and $CDA$. Which triangle is $y$ located inside?

# %%
# Function to compute barycentric coordinates
def barycentric_coords(A, B, C, P):
    denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    r1 = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denom
    r2 = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denom
    r3 = 1 - r1 - r2
    return r1, r2, r3

# Sample data
rng = np.random.default_rng(2024)
X = rng.uniform(size=(50, 2))
y = rng.uniform(size=(2,))

# Defining points A, B, C, D
A = min([x for x in X if x[0] > y[0] and x[1] > y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
B = min([x for x in X if x[0] > y[0] and x[1] < y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
C = min([x for x in X if x[0] < y[0] and x[1] < y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
D = min([x for x in X if x[0] < y[0] and x[1] > y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))

# Compute barycentric coordinates for point y with respect to triangles ABC and CDA
rABC1, rABC2, rABC3 = barycentric_coords(A, B, C, y)
rCDA1, rCDA2, rCDA3 = barycentric_coords(C, D, A, y)

# Check if the point is inside the triangles
def is_inside_triangle(r1, r2, r3):
    return (r1 >= 0) and (r2 >= 0) and (r3 >= 0)

inside_ABC = is_inside_triangle(rABC1, rABC2, rABC3)
inside_CDA = is_inside_triangle(rCDA1, rCDA2, rCDA3)

# Plotting
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Random Points')
plt.scatter(*y, color='red', label='Point y', zorder=5)
plt.scatter(*A, color='green', label='Point A', zorder=5)
plt.scatter(*B, color='green', label='Point B', zorder=5)
plt.scatter(*C, color='green', label='Point C', zorder=5)
plt.scatter(*D, color='green', label='Point D', zorder=5)

plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color='black', label='Triangle ABC')
plt.plot([C[0], D[0], A[0], C[0]], [C[1], D[1], A[1], C[1]], color='purple', label='Triangle CDA')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Barycentric Interpolation Visualization')
plt.grid(True)
plt.show()

# Print results
print("Point y:", y)
print("Triangle ABC: A =", A, ", B =", B, ", C =", C)
print("Triangle CDA: C =", C, ", D =", D, ", A =", A)
print("Barycentric coordinates with respect to ABC:", rABC1, rABC2, rABC3)
print("Barycentric coordinates with respect to CDA:", rCDA1, rCDA2, rCDA3)
print("Is point y inside triangle ABC?", inside_ABC)
print("Is point y inside triangle CDA?", inside_CDA)

# %% [markdown]
# As seen in both the figure and output, point y is located inside triangle ABC

# %% [markdown]
# Now consider the function:
# $$
# f(x_1,x_2) = x_1 \cdot x_2
# $$

# %%
f = lambda x: x[0]*x[1]
F = np.array([f(x) for x in X])

# %% [markdown]
# **Question 3:** Compute the approximation of $f(y)$ using the full algorithm. Compare with the true value.

# %%
# Function to compute barycentric coordinates
def barycentric_coords(A, B, C, P):
    denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    r1 = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denom
    r2 = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denom
    r3 = 1 - r1 - r2
    return r1, r2, r3

# Function to interpolate the value at point y using barycentric coordinates
def interpolate_value(A, B, C, r1, r2, r3, f):
    return r1 * f(A) + r2 * f(B) + r3 * f(C)

# Sample data
rng = np.random.default_rng(2024)
X = rng.uniform(size=(50, 2))
y = rng.uniform(size=(2,))

# True function
f = lambda x: x[0] * x[1]

# Find points A, B, C, D
A = min([x for x in X if x[0] > y[0] and x[1] > y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
B = min([x for x in X if x[0] > y[0] and x[1] < y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
C = min([x for x in X if x[0] < y[0] and x[1] < y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
D = min([x for x in X if x[0] < y[0] and x[1] > y[1]], key=lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))

# Compute barycentric coordinates for point y with respect to triangles ABC and CDA
rABC1, rABC2, rABC3 = barycentric_coords(A, B, C, y)
rCDA1, rCDA2, rCDA3 = barycentric_coords(C, D, A, y)

# Check if the point is inside the triangles
def is_inside_triangle(r1, r2, r3):
    return (r1 >= 0) and (r2 >= 0) and (r3 >= 0)

inside_ABC = is_inside_triangle(rABC1, rABC2, rABC3)
inside_CDA = is_inside_triangle(rCDA1, rCDA2, rCDA3)

# Interpolate the function value at point y
if inside_ABC:
    interpolated_value = interpolate_value(A, B, C, rABC1, rABC2, rABC3, f)
elif inside_CDA:
    interpolated_value = interpolate_value(C, D, A, rCDA1, rCDA2, rCDA3, f)
else:
    interpolated_value = None

# True value at point y
true_value = f(y)

# Print results
print("Point y:", y)
print("Triangle ABC: A =", A, ", B =", B, ", C =", C)
print("Triangle CDA: C =", C, ", D =", D, ", A =", A)
print("Barycentric coordinates with respect to ABC:", rABC1, rABC2, rABC3)
print("Barycentric coordinates with respect to CDA:", rCDA1, rCDA2, rCDA3)
print("Is point y inside triangle ABC?", inside_ABC)
print("Is point y inside triangle CDA?", inside_CDA)
print("Interpolated value at point y:", interpolated_value)
print("True value at point y:", true_value)

# %% [markdown]
# The approximation of f(y) is shown as the "Interpolated value at point y," which is 0.08405201731052576. This value is obtained using the barycentric interpolation method.The approximation (interpolated value) 
# is 0.08405201731052576, and the true value of the function at 
# y is 0.0789565216259594. The slight difference between these values reflects the interpolation error.

# %% [markdown]
# **Question 4:** Repeat question 3 for all points in the set $Y$.

# %%
Y = [(0.2,0.2),(0.8,0.2),(0.8,0.8),(0.8,0.2),(0.5,0.5)]

# %%
# Function to compute barycentric coordinates
def barycentric_coords(A, B, C, P):
    denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    if abs(denom) < 1e-10:  # Handle division by zero
        return None, None, None
    r1 = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denom
    r2 = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denom
    r3 = 1 - r1 - r2
    return r1, r2, r3

# Function to interpolate the value at point y using barycentric coordinates
def interpolate_value(A, B, C, r1, r2, r3, f):
    return r1 * f(A) + r2 * f(B) + r3 * f(C)

# Sample data
rng = np.random.default_rng(2024)
X = rng.uniform(size=(50, 2))

# True function
f = lambda x: x[0] * x[1]

# Points to interpolate
Y = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.8, 0.2), (0.5, 0.5)]

# Store results
interpolated_values = []
true_values = []

def find_points(X, y):
    A = min(X, key=lambda x: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2))
    B = min([x for x in X if x[0] > y[0]], key=lambda x: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2), default=A)
    C = min([x for x in X if x[1] < y[1]], key=lambda x: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2), default=A)
    D = min([x for x in X if x[0] < y[0]], key=lambda x: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2), default=A)
    return A, B, C, D

for y in Y:
    A, B, C, D = find_points(X, y)
    
    rABC1, rABC2, rABC3 = barycentric_coords(A, B, C, y)
    rCDA1, rCDA2, rCDA3 = barycentric_coords(C, D, A, y)
    
    def is_inside_triangle(r1, r2, r3):
        if r1 is None or r2 is None or r3 is None:
            return False
        return (r1 >= 0) and (r2 >= 0) and (r3 >= 0)
    
    inside_ABC = is_inside_triangle(rABC1, rABC2, rABC3)
    inside_CDA = is_inside_triangle(rCDA1, rCDA2, rCDA3)
    
    if inside_ABC:
        interpolated_value = interpolate_value(A, B, C, rABC1, rABC2, rABC3, f)
    elif inside_CDA:
        interpolated_value = interpolate_value(C, D, A, rCDA1, rCDA2, rCDA3, f)
    else:
        interpolated_value = None

    true_value = f(y)
    
    interpolated_values.append(interpolated_value)
    true_values.append(true_value)

# Print results
for i, y in enumerate(Y):
    print(f"Point y: {y}")
    print(f"Interpolated value at point y: {interpolated_values[i]}")
    print(f"True value at point y: {true_values[i]}")
    print("-" * 40)

# %% [markdown]
# We had som trouble retrieving interpolated values for most of the point, but as seen from point (0.5,0.5) the interpolated value is very close to the true value. This was also the case in question 3, which could indicate that it would be the same for the other values.


