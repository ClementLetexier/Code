#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:53:15 2025

@author: clement
"""
# Libraries

import numpy as np 
from scipy.stats import norm 
import matplotlib.pyplot as plt
import qfin as qf

# Option Parameters 

S = 100                 
K = 105
vol = 0.15
r = 0.05
T = 1
N = 252
dt = T / N

# Call price computation

def call_price(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + vol ** 2 / 2) * T) / (vol * np.sqrt(T))  
    d2 = d1 - vol * np.sqrt(T)  
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


# Simulated path of the underlying asset

def Geometric_Brownian_Motion(S, r, vol, T, dt, N_paths) : 
    N_steps = int(T / dt) # Sets the number of steps for which the GBM will give a value
    t = np.linspace(0, N_steps, N_steps) # np.linspace creates an array of time points, from 0 to N_steps, divided into N_steps intervals
    Si = np.zeros((N_paths, N_steps)) # Creates an empty matrix to store each data point for each combination of time step and path
    Si[:,0] = S # Sets the price at the beginning of the simulation as S
    
    for i in range(1, N_steps) : # the for loop starts at 1 because 0 is already set as S
        Z = np.random.normal(0,1,N_paths) # Sets a random value representing market randomness
        Si[:, i] = Si[:, i-1] * np.exp((r - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z) # Geometric Brownian Motion Formula
        
    return t, Si # Imagine a 252 part timeframe which gives you the day, set above a table which gives you the price for each day, on each simulated path
    

# Simulation Parameters

N_paths = 1

# Running the Simulation of the Path

t, Si = Geometric_Brownian_Motion(S, r, vol, T, dt, N_paths)

# Plotting the simulated path and performance of the call option

plt.figure(figsize = (10, 5))
plt.hlines(105,0,252, label = 'Strike Price', color = 'orange')
plt.xlabel('Days')
plt.ylabel('Asset Price')
plt.style.use('dark_background')

for i in range(N_paths) :
    plt.plot(t, Si[i,:], linewidth = 1, color = 'white', label = 'Asset Price')
    
plt.fill_between(t, Si[i, :], 105, where=(Si[i, :] < 105), color='red', alpha=0.3, label = 'Negative Performance') 
plt.fill_between(t, Si[i, :], 105, where=(Si[i, :] > 105), color='green', alpha=0.3, label = 'Positive Performance')
plt.grid(alpha = 0.2, zorder = 0)
plt.legend()
plt.show()

# Trading Strategy

print('Model Price is : ', round(call_price(S, K, T, r, vol), 2))
print('Market Price is : ', round(call_price(S, K, T, r, vol),2) - 0.2)
print('Market Discrepancy is : ', 0.2)

# Computing the P&L of buying 100 such options N times

premium = 5.84 * 100
pnls = []

for i in range(100000):  # Define N number of simulations
    _, Si = Geometric_Brownian_Motion(S, r, vol, T, dt, N_paths)  # Simulate N paths
    
    final_price = Si[:, -1]  # Get the final stock price for each path
    option_payout = np.maximum(final_price - K, 0) * 100   # Call option payoff (max(ST - K, 0) * 100 contracts)
    
    pnl = option_payout - premium  # Profit/Loss calculation
    pnls.append(pnl)

# Compute the average P&L
expected_pnl = np.mean(pnls)
print("Expected P&L:", expected_pnl) # Strategy has a positive expectancy

# Create a plot for Equity curve of strategy

equity = np.cumsum(pnls)
plt.figure(figsize = (10,5))
plt.plot(equity, linewidth = 1, color = 'darkorange', label = 'Equity Curve')
plt.xlabel('Amount of Trades')
plt.ylabel('Equity Value')
plt.grid(alpha = 0.2, zorder = 0)
plt.legend()
plt.show()


### Add proof of positive expenctency before plot

