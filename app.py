import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Blurb at the top explaining the purpose of the app
st.title("Stochastic Process Simulator")

st.write("""
This app simulates and visualizes common stochastic processes used in financial models. 
You can choose from Geometric Brownian Motion (GBM), which is commonly used to model stock prices, 
or the Ornstein-Uhlenbeck Process, which is used for modeling mean-reverting processes like interest rates or volatility.
""")

# Sidebar for user inputs
st.sidebar.header("Simulation Parameters")

# Stochastic process selection
process_type = st.sidebar.selectbox('Select Stochastic Process', options=['Geometric Brownian Motion (GBM)', 'Ornstein-Uhlenbeck Process'])

# Common parameters for both processes
time_horizon = st.sidebar.slider('Time Horizon (T)', min_value=1, max_value=365, value=100)
n_steps = st.sidebar.slider('Number of Steps (N)', min_value=10, max_value=1000, value=100)
n_paths = st.sidebar.slider('Number of Paths', min_value=1, max_value=10, value=5)

# Parameters specific to Geometric Brownian Motion
if process_type == 'Geometric Brownian Motion (GBM)':
    st.sidebar.subheader("GBM Parameters")
    mu = st.sidebar.slider('Drift (μ)', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    sigma = st.sidebar.slider('Volatility (σ)', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    initial_value = st.sidebar.number_input('Initial Value (S0)', value=100.0)

# Parameters specific to Ornstein-Uhlenbeck Process
elif process_type == 'Ornstein-Uhlenbeck Process':
    st.sidebar.subheader("Ornstein-Uhlenbeck Parameters")
    theta = st.sidebar.slider('Speed of Mean Reversion (θ)', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    mu_ou = st.sidebar.slider('Long-term Mean (μ)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    sigma_ou = st.sidebar.slider('Volatility (σ)', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    initial_value = st.sidebar.number_input('Initial Value (X0)', value=0.0)

# Function to simulate Geometric Brownian Motion (GBM)
def simulate_gbm(mu, sigma, S0, T, N, n_paths):
    dt = T / N
    time_grid = np.linspace(0, T, N)
    paths = np.zeros((N, n_paths))
    paths[0] = S0
    for i in range(1, N):
        z = np.random.standard_normal(n_paths)
        paths[i] = paths[i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return time_grid, paths

# Function to simulate Ornstein-Uhlenbeck Process
def simulate_ou(theta, mu, sigma, X0, T, N, n_paths):
    dt = T / N
    time_grid = np.linspace(0, T, N)
    paths = np.zeros((N, n_paths))
    paths[0] = X0
    for i in range(1, N):
        z = np.random.standard_normal(n_paths)
        paths[i] = paths[i - 1] + theta * (mu - paths[i - 1]) * dt + sigma * np.sqrt(dt) * z
    return time_grid, paths

# Simulation and plotting
st.subheader(f"Simulating {process_type}")

if process_type == 'Geometric Brownian Motion (GBM)':
    time_grid, paths = simulate_gbm(mu, sigma, initial_value, time_horizon, n_steps, n_paths)
elif process_type == 'Ornstein-Uhlenbeck Process':
    time_grid, paths = simulate_ou(theta, mu_ou, sigma_ou, initial_value, time_horizon, n_steps, n_paths)

# Plot the simulated paths
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(n_paths):
    ax.plot(time_grid, paths[:, i], lw=1.5)
ax.set_title(f"Simulated Paths for {process_type}")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
st.pyplot(fig)

# Summary of simulation parameters
st.write(f"""
### Summary of {process_type} Simulation
- **Time Horizon (T)**: {time_horizon}
- **Number of Steps (N)**: {n_steps}
- **Number of Simulated Paths**: {n_paths}
""")

# Specific summary for each process
if process_type == 'Geometric Brownian Motion (GBM)':
    st.write(f"""
    - **Drift (μ)**: {mu}
    - **Volatility (σ)**: {sigma}
    - **Initial Value (S0)**: {initial_value}
    """)
elif process_type == 'Ornstein-Uhlenbeck Process':
    st.write(f"""
    - **Speed of Mean Reversion (θ)**: {theta}
    - **Long-term Mean (μ)**: {mu_ou}
    - **Volatility (σ)**: {sigma_ou}
    - **Initial Value (X0)**: {initial_value}
    """)

st.write("""
### Stochastic Process Overview:
- **Geometric Brownian Motion (GBM)** is used in the Black-Scholes model for stock price modeling. It assumes a constant drift and volatility, which leads to an exponential growth pattern in the long run.
- **Ornstein-Uhlenbeck Process** is a mean-reverting stochastic process, often used to model interest rates and volatility. It tends to pull the simulated values back to the long-term mean (μ) over time.
""")
