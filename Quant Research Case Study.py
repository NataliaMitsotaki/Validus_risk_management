"""

Parameters

s0 - initial stock price
r - risk free interest rate
T - time of maturity (in years)
N - number of periods until maturity
K - strike price
q - risk neutral probability that the stock will move upwards
v - return the stock will provide with probability q (upward move)

"""

import numpy as np
from scipy import optimize
import itertools

## Question 1 ##

def european_call(v, K, N, T, s0=1, r=0):
    
    # Calculate risk-neutral probability (q=0.5 because r=0)
    u = 1+v
    d = 1-v
    R = np.exp(r * T/N)
    p = (R - d) / (u - d)
    
    
    # Generate binomial price tree
    price_tree = np.zeros([N+1, N+1])
    
    for i in range(N+1):
        for j in range(i, N+1):
            price_tree[i, j] = s0 * u**(j-i) * d**i
    
    
    # Calculate expected call option payoffs at maturity
    payoffs = np.zeros(N+1)
    for i in range(N+1):
        payoffs[i] = max(0, price_tree[i, N] - K)
    
    
    # Work backwards in the tree to calculate option value at each node
    option_values = np.zeros([N+1, N+1])
    option_values[:, N] = payoffs

    for j in range(N-1, -1, -1):
        for i in range(j+1):
            option_values[i, j] = (1/R) * (p * option_values[i, j+1] + (1-p) * option_values[i+1, j+1])
    
    # Return the option value at time t=0
    V = option_values[0,0]
    
    return V


## Question 2 ##
    
def move_calibration(V, K, N, T, s0=1, r=0):
    
    # Define the objective function of the optimisation problem
    def option_value_diff(v0, V, K, N, T, s0, r):
    
        model_value = european_call(v0, K, N, T, s0, r)
    
        diff = abs(V - model_value)
        
        return diff
    
    # Initialise the calibration parameter
    v0 = 0.5
    
    # Minimize option_value_diff function with respect to v
    result = optimize.fmin(option_value_diff, x0 = v0, args = (V, K, N, T, s0, r))
    
    # Return the value v that minimises the objective function
    v_hat = result[0]
    
    return v_hat


## Question 3 ##

def american_call(v, K, N, T, s0=1, r=0):
    
    # Calculate risk-neutral probability (q=0.5 because r=0)
    u = 1+v
    d = 1-v
    R = np.exp(r * T/N)
    p = (R - d) / (u - d)
    
    
    # Generate binomial price tree
    price_tree = np.zeros([N+1, N+1])
    
    for i in range(N+1):
        for j in range(i, N+1):
            price_tree[i, j] = s0 * u**(j-i) * d**i
    
    
    # Calculate expected call option payoffs at maturity
    payoffs = np.zeros(N+1)
    for i in range(N+1):
        payoffs[i] = max(0, price_tree[i, N] - K)
    
    
    # Work backwards in the tree to calculate option value at each node
    option_values = np.zeros([N+1, N+1])
    option_values[:, N] = payoffs
    
    ## in an American option, we check at each node if it is optimal to exercise
    time_exercise = N

    for j in range(N-1, -1, -1):
        for i in range(j+1):
            option_values[i, j] = max((1/R) * (p * option_values[i, j+1] + (1-p) * option_values[i+1, j+1]), price_tree[i, j]-K)
            if option_values[i, j] == price_tree[i, j]-K and j < time_exercise:
                time_exercise = j
    
    # Return the option value at time t=0
    V = option_values[0,0]
    
    print("Optimal to exercise at time t =", time_exercise)
    
    return V


## Question 4 ##

def expected_max(v, K, N, T, s0=1, r=0):
    
    # Calculate risk-neutral probability (q=0.5 because r=0)
    u = 1+v
    d = 1-v
    R = np.exp(r * T/N)
    p = (R - d) / (u - d)
    
    # Generate binomial price tree
    price_tree = np.zeros([N+1, N+1])
    
    for i in range(N+1):
        for j in range(i, N+1):
            price_tree[i, j] = s0 * u**(j-i) * d**i
    
    # Calculate the probability of occurrence of each path of the binomial tree
    moves = [1, 0]
    paths = np.array(list(itertools.product(moves, repeat=N)))
    upwards = np.sum(paths, axis=1)
    prob_path = [p**u * (1-p)**(N-u) for u in upwards]

    # check if there has been any error in the prob_path creation
    if sum(prob_path) != 1:
        print("Check mistake")


    # Create an array with all the possible paths
    path_array = np.zeros([2**N, N+1])
    
    # Each path starts from the initial stock price
    path_array[:,0] = s0
    
    for i in range(2**N):
        row = 0
        for j in range(1, N+1):
            
            # in case there is an upward move in the stock price
            if paths[i][j-1] == 1:
                path_array[i, j] = price_tree[row, j]
            
            # in case there is a downward move in the stock price
            else:
                row += 1
                path_array[i, j] = price_tree[row, j]

    
    # Calculate the max stock price of each possible path
    max_path = np.amax(path_array, axis=1)
    
    # Calculate expectation of maximum as the sum of (max values * probabilities)
    expected_max = np.sum(np.multiply(max_path, prob_path))

    return expected_max
                
                
            