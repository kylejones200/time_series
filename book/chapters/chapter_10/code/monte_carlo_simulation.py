import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo_simulation(initial_value, mu, sigma, days, simulations=1000):
    """Perform a Monte Carlo simulation for stock price forecasting."""
    dt = 1  # Daily step size
    prices = np.zeros((days, simulations))
    prices[0] = initial_value
    
    for t in range(1, days):
        random_shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=simulations)
        prices[t] = prices[t - 1] * np.exp(random_shock)
    
    return prices

def plot_monte_carlo(prices, title="Monte Carlo Simulation"):
    """Plot the Monte Carlo simulation results."""
    plt.figure(figsize=(10, 5))
    plt.plot(prices, color="gray", alpha=0.3)
    plt.plot(prices.mean(axis=1), color="black", linewidth=2, label="Mean Path")
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Simulated Price")
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    simulated_prices = monte_carlo_simulation(initial_value=100, mu=0.0005, sigma=0.02, days=252)
    plot_monte_carlo(simulated_prices, "Stock Price Monte Carlo Simulation")

