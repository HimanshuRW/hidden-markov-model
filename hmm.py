import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
data = pd.read_csv("stock.csv")
# date,                   price,    feature1, feature2, ..., featureN
# 2024-08-01 12:01:00,    307.7,    0.5,      -0.2,      ..., 1.3
# 2024-08-01 12:02:00,    308.1,    0.6,      -0.1,      ..., 1.1
# ...

# Select key features for HMM (excluding date and price for features)
features = ['feature1', 'feature2', 'feature3',..., 'featureN']

# Prepare feature matrix
X = data[features].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🤖 HMM Trading Agent Initialization")
print("="*50)
print(f"Dataset size: {len(data)} points")
print(f"Features used: {len(features)}")
print(f"Training HMM with 3 market regimes...")

# Initialize and train HMM with 3 states (Bear, Neutral, Bull)
model = hmm.GaussianHMM(n_components=3, covariance_type="full", random_state=42)
model.fit(X_scaled)

# Predict hidden states
hidden_states = model.predict(X_scaled)

# Add states to dataframe
data['regime'] = hidden_states
data['price_returns'] = data['price'].pct_change()

print("\n📊 Market Regime Analysis")
print("="*50)

# Analyze each regime
regime_stats = {}
for state in range(3):
    mask = data['regime'] == state
    avg_return = data.loc[mask, 'price_returns'].mean()
    volatility = data.loc[mask, 'price_returns'].std()
    count = mask.sum()
    
    regime_stats[state] = {
        'avg_return': avg_return,
        'volatility': volatility,
        'count': count,
        'percentage': count / len(data) * 100
    }
    
    print(f"Regime {state}: {count:5d} points ({count/len(data)*100:.1f}%)")
    print(f"  Avg Return: {avg_return:8.6f}")
    print(f"  Volatility: {volatility:8.6f}")
    print(f"  Risk/Reward: {avg_return/volatility if volatility > 0 else 0:.4f}")

# Classify regimes based on returns
regime_names = {}
sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['avg_return'])
regime_names[sorted_regimes[0][0]] = "BEAR 🐻"
regime_names[sorted_regimes[1][0]] = "NEUTRAL 🦘" 
regime_names[sorted_regimes[2][0]] = "BULL 🐂"

print(f"\n🎯 Regime Classification:")
for state, name in regime_names.items():
    print(f"State {state} = {name}")

# Generate trading signals
def generate_signals(data, regime_names):
    signals = []
    positions = []
    current_position = 0  # 0: no position, 1: long, -1: short
    
    for i, row in data.iterrows():
        regime = row['regime']
        regime_name = regime_names[regime]
        
        # Trading logic based on regime
        if "BULL" in regime_name:
            signal = "BUY" if current_position <= 0 else "HOLD_LONG"
            current_position = 1
        elif "BEAR" in regime_name:
            signal = "SELL" if current_position >= 0 else "HOLD_SHORT"
            current_position = -1
        else:  # NEUTRAL
            signal = "NEUTRAL"
            # Keep current position in neutral regime
        
        signals.append(signal)
        positions.append(current_position)
    
    return signals, positions

# Generate signals
signals, positions = generate_signals(data, regime_names)
data['signal'] = signals
data['position'] = positions

print(f"\n📈 Trading Signals Generated")
print("="*50)

signal_counts = data['signal'].value_counts()
for signal, count in signal_counts.items():
    print(f"{signal:12}: {count:5d} ({count/len(data)*100:.1f}%)")

# Calculate strategy performance
data['strategy_returns'] = data['position'].shift(1) * data['price_returns']
data['cumulative_returns'] = (1 + data['price_returns']).cumprod()
data['cumulative_strategy'] = (1 + data['strategy_returns'].fillna(0)).cumprod()

# Performance metrics
total_return = data['cumulative_strategy'].iloc[-1] - 1
buy_hold_return = data['cumulative_returns'].iloc[-1] - 1
strategy_vol = data['strategy_returns'].std()
sharpe_ratio = data['strategy_returns'].mean() / strategy_vol if strategy_vol > 0 else 0

print(f"\n💰 Performance Summary")
print("="*50)
print(f"Strategy Return:    {total_return:8.4f} ({total_return*100:.2f}%)")
print(f"Buy & Hold Return:  {buy_hold_return:8.4f} ({buy_hold_return*100:.2f}%)")
print(f"Excess Return:      {total_return - buy_hold_return:8.4f}")
print(f"Strategy Volatility: {strategy_vol:8.6f}")
print(f"Sharpe Ratio:       {sharpe_ratio:8.4f}")

# Recent signals (last 20)
print(f"\n🔥 Recent Trading Signals (Last 20)")
print("="*60)
recent_data = data.tail(20)[['price', 'regime', 'signal', 'position']].copy()
recent_data['regime_name'] = recent_data['regime'].map(regime_names)

for i, row in recent_data.iterrows():
    print(f"Price: {row['price']:7.2f} | {row['regime_name']:12} | {row['signal']:12} | Pos: {row['position']:2d}")

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Price and regimes
colors = ['red', 'orange', 'green']
sample_indices = np.arange(0, len(data), max(1, len(data)//1000))  # Sample for plotting
sample_data = data.iloc[sample_indices]

scatter = ax1.scatter(range(len(sample_data)), sample_data['price'], 
                     c=[colors[regime] for regime in sample_data['regime']], 
                     alpha=0.6, s=1)
ax1.set_title('Price Colored by Market Regime')
ax1.set_ylabel('Price')
ax1.legend(['Bear 🐻', 'Neutral 🦘', 'Bull 🐂'])

# Cumulative returns comparison
ax2.plot(data['cumulative_returns'].values, label='Buy & Hold', linewidth=1)
ax2.plot(data['cumulative_strategy'].values, label='HMM Strategy', linewidth=1)
ax2.set_title('Cumulative Returns Comparison')
ax2.set_ylabel('Cumulative Return')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Regime distribution
regime_dist = [regime_stats[i]['count'] for i in range(3)]
regime_labels = [regime_names[i] for i in range(3)]
ax3.pie(regime_dist, labels=regime_labels, autopct='%1.1f%%', startangle=90)
ax3.set_title('Market Regime Distribution')

# Feature importance (based on regime means)
feature_importance = np.abs(model.means_).mean(axis=0)
ax4.barh(features, feature_importance)
ax4.set_title('Feature Importance in Regime Detection')
ax4.set_xlabel('Importance')

plt.tight_layout()
plt.show()

print(f"\n🎉 HMM Trading Agent Complete!")
print(f"Model identifies {len(set(hidden_states))} market regimes")
print(f"Strategy generated {len([s for s in signals if s in ['BUY', 'SELL']])} trading signals")