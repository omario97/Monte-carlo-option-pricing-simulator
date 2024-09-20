import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Option Pricing Simulator", page_icon="📊")  # or "OptionSense" for the other app

st.title("Monte Carlo Option Pricing Simulator")  # or "OptionSense: Dynamic Option Pricing Sensitivity Analyzer" for the other app

st.markdown("*Source code available on my [GitHub](https://github.com/omario97) page*")
# Add creator info and social links to the sidebar
st.sidebar.markdown("<h3 style='text-align: left; color: #888888;'>Created by Omar Hussain</h3>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="display: flex; justify-content: left; align-items: center;">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <a href="https://www.linkedin.com/in/omar-hussain-504777164/" target="_blank" style="text-decoration: none; color: inherit; margin-right: 20px;">
        <i class="fab fa-linkedin" style="font-size: 24px;"></i>
    </a>
    <a href="https://github.com/omario97" target="_blank" style="text-decoration: none; color: inherit;">
        <i class="fab fa-github" style="font-size: 24px;"></i>
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

def black_scholes(S0, K, T, r, sigma, option_type):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def monte_carlo_option_pricing(S0, K, T, r, sigma, option_type, num_simulations=10000, num_steps=252, num_batches=10):
    batch_size = num_simulations // num_batches
    batch_prices = np.zeros(num_batches)
    
    for i in range(num_batches):
        dt = T / num_steps
        nudt = (r - 0.5 * sigma**2) * dt
        sigdt = sigma * np.sqrt(dt)
        
        S = S0 * np.exp(np.cumsum(nudt + sigdt * np.random.standard_normal((batch_size, num_steps)), axis=1))
        
        if option_type == 'call':
            payoff = np.maximum(S[:, -1] - K, 0)
        else:  # put
            payoff = np.maximum(K - S[:, -1], 0)
        
        batch_prices[i] = np.exp(-r * T) * np.mean(payoff)
    
    option_price = np.mean(batch_prices)
    price_std = np.std(batch_prices)
    
    return option_price, price_std

def monte_carlo_antithetic(S0, K, T, r, sigma, option_type, num_simulations=5000, num_steps=100, num_batches=10):
    batch_size = num_simulations // num_batches
    batch_prices = np.zeros(num_batches)
    
    for i in range(num_batches):
        dt = T / num_steps
        nudt = (r - 0.5 * sigma**2) * dt
        sigdt = sigma * np.sqrt(dt)
        
        Z = np.random.standard_normal((batch_size, num_steps))
        S_up = S0 * np.exp(np.cumsum(nudt + sigdt * Z, axis=1))
        S_down = S0 * np.exp(np.cumsum(nudt - sigdt * Z, axis=1))
        
        if option_type == 'call':
            payoff_up = np.maximum(S_up[:, -1] - K, 0)
            payoff_down = np.maximum(S_down[:, -1] - K, 0)
        else:  # put
            payoff_up = np.maximum(K - S_up[:, -1], 0)
            payoff_down = np.maximum(K - S_down[:, -1], 0)
        
        batch_prices[i] = np.exp(-r * T) * np.mean((payoff_up + payoff_down) / 2)
    
    option_price = np.mean(batch_prices)
    price_std = np.std(batch_prices)
    
    return option_price, price_std

def monte_carlo_control_variate(S0, K, T, r, sigma, option_type, num_simulations=10000, num_steps=100, num_batches=10):
    batch_size = num_simulations // num_batches
    batch_prices = np.zeros(num_batches)
    
    for i in range(num_batches):
        dt = T / num_steps
        nudt = (r - 0.5 * sigma**2) * dt
        sigdt = sigma * np.sqrt(dt)
        
        Z = np.random.standard_normal((batch_size, num_steps))
        S = S0 * np.exp(np.cumsum(nudt + sigdt * Z, axis=1))
        
        if option_type == 'call':
            payoff = np.maximum(S[:, -1] - K, 0)
        else:  # put
            payoff = np.maximum(K - S[:, -1], 0)
        
        # Control variate
        Y = S[:, -1]  # Stock price at maturity
        EY = S0 * np.exp(r * T)  # Expected stock price at maturity
        
        cov_matrix = np.cov(payoff, Y)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        controlled_payoff = payoff - beta * (Y - EY)
        batch_prices[i] = np.exp(-r * T) * np.mean(controlled_payoff)
    
    option_price = np.mean(batch_prices)
    price_std = np.std(batch_prices)
    
    return option_price, price_std

def calculate_option_prices(S0, K, T, r, sigma, option_type):
    bs_price = black_scholes(S0, K, T, r, sigma, option_type)
    mc_price, mc_std = monte_carlo_option_pricing(S0, K, T, r, sigma, option_type)
    return bs_price, mc_price, mc_std


# Streamlit app
st.title("Interactive Option Pricing with Monte Carlo Simulations")

# User inputs
st.sidebar.header("Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", min_value=1.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T) in years", min_value=1, value=1, step=1)
r = st.sidebar.number_input("Risk-free Rate (r)", min_value=0.0, value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1000, value=10000, step=1000)
num_batches = st.sidebar.number_input("Number of Batches", min_value=1, value=10, step=1)

st.write(f"Pricing a {option_type} option with {num_simulations} simulations, divided into {num_batches} batches.")

# Calculate prices using all methods
bs_price = black_scholes(S0, K, T, r, sigma, option_type)
mc_price, mc_std = monte_carlo_option_pricing(S0, K, T, r, sigma, option_type, num_simulations, num_batches=num_batches)
anti_price, anti_std = monte_carlo_antithetic(S0, K, T, r, sigma, option_type, num_simulations//2, num_batches=num_batches)
cv_price, cv_std = monte_carlo_control_variate(S0, K, T, r, sigma, option_type, num_simulations, num_batches=num_batches)

# Create a dataframe for comparison
df = pd.DataFrame({
    'Method': ['Black-Scholes', 'Standard MC', 'Antithetic Variables', 'Control Variate'],
    'Price': [bs_price, mc_price, anti_price, cv_price],
    'Std Dev': [0, mc_std, anti_std, cv_std]
})

st.subheader("Comparison of Option Pricing Methods")
st.table(df.style.format({'Price': '${:.4f}', 'Std Dev': '${:.4f}'}))

# Visualization of Standard Deviation only
fig = go.Figure()
fig.add_trace(go.Bar(x=df['Method'], y=df['Std Dev'], name='Standard Deviation'))

fig.update_layout(
    title=f"{option_type.capitalize()} Option Price Standard Deviation Comparison",
    xaxis_title="Method",
    yaxis_title="Standard Deviation",
    showlegend=False
)

st.plotly_chart(fig)

# Detailed explanation of methods
st.header("Explanation of Methods")

st.subheader("1. Black-Scholes Model")
st.write("The Black-Scholes model is an analytical solution for option pricing. It provides a benchmark for comparison with Monte Carlo methods.")

st.subheader("2. Standard Monte Carlo")
st.write("This method simulates multiple price paths and averages the payoffs to estimate the option price. It's versatile but can be computationally intensive for high precision.")

st.subheader("3. Antithetic Variables")
st.write("This variance reduction technique uses pairs of negatively correlated random numbers. It often improves precision without increasing computational cost significantly.")

st.subheader("4. Control Variate")
st.write("This method uses known information about a related variable (stock price at maturity) to reduce variance in the option price estimate. It typically provides the highest precision but may require more computation.")

# Add an interactive feature to compare methods
selected_methods = st.multiselect(
    "Select methods to compare",
    ["Standard MC", "Antithetic Variables", "Control Variate"],
    default=["Standard MC", "Antithetic Variables", "Control Variate"]
)

if selected_methods:
    comparison_df = df[df['Method'].isin(selected_methods + ['Black-Scholes'])]
    st.subheader("Comparison of Selected Methods")
    st.table(comparison_df.style.format({'Price': '${:.4f}', 'Std Dev': '${:.4f}'}))

    # Calculate and display percentage differences
    for method in selected_methods:
        diff = (comparison_df[comparison_df['Method'] == method]['Price'].values[0] - bs_price) / bs_price * 100
        st.write(f"{method} differs from Black-Scholes by {diff:.2f}%")

# Add a feature to run multiple simulations and show distribution
if st.button("Run Multiple Simulations"):
    num_runs = 100
    results = []
    for _ in range(num_runs):
        _, price, _ = calculate_option_prices(S0, K, T, r, sigma, option_type)
        results.append(price)
    
    fig = go.Figure(data=[go.Histogram(x=results)])
    fig.update_layout(title=f"Distribution of {option_type.capitalize()} Option Prices (100 runs)")
    st.plotly_chart(fig)
    st.write(f"Mean: ${np.mean(results):.4f}, Std Dev: ${np.std(results):.4f}")


