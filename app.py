# ==============================================================================
# FINAL PROJECT: app.py (Version 25 - Definitive Logic & UI Optimization)
#
# This version corrects the faulty sensor reputation logic and optimizes the
# simulation speed to create a smooth, flicker-free UI experience. This is the
# definitive, feature-complete application.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import logging
import hashlib
import sys
from io import StringIO
import contextlib

# --- ML & DL Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector

# ==============================================================================
# 0. LOGGING SETUP
# ==============================================================================
# Configure logger to write to a file, overwriting on each new run.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename='live_simulation_log.txt',
    filemode='w'
)
logging.info("--- New Simulation Run Started ---")

# ==============================================================================
# 1. LIGHTWEIGHT BLOCKCHAIN IMPLEMENTATION
# ==============================================================================
class Blockchain:
    """
    A lightweight, WSN-friendly "Micro-Ledger" implemented in Python.
    It manages sensor reputations using a simple, cryptographically-chained ledger
    without the overhead of complex consensus algorithms.
    """
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.sensor_reputations = {}
        # Create the genesis block (the first block in the chain)
        self.create_block(previous_hash='0', nonce=100)

    def create_block(self, nonce, previous_hash):
        """Creates a new block and adds it to the chain."""
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': self.pending_transactions,
            'nonce': nonce,
            'previous_hash': previous_hash,
        }
        self.pending_transactions = []
        self.chain.append(block)
        return block

    @staticmethod
    def hash(block):
        """Creates a SHA-256 hash of a block."""
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def get_last_block(self):
        """Returns the most recent block in the chain."""
        return self.chain[-1]

    def add_transaction_and_mine(self, transaction):
        """Adds a new transaction and 'mines' a new block to store it."""
        self.pending_transactions.append(transaction)
        last_block = self.get_last_block()
        previous_hash = self.hash(last_block)
        new_block = self.create_block(nonce=last_block['nonce'] + 1, previous_hash=previous_hash)
        return new_block['index']

    def update_reputation(self, sensor_id, is_true_positive):
        """
        Updates a sensor's reputation based on its contribution to an alert
        and creates a transaction for this event.
        """
        # Initialize sensor if it's the first time we've seen it.
        if sensor_id not in self.sensor_reputations:
            self.sensor_reputations[sensor_id] = {'reputation': 100, 'status': 'Active'}

        # Only update active sensors.
        if self.sensor_reputations[sensor_id]['status'] == 'Active':
            if is_true_positive:
                self.sensor_reputations[sensor_id]['reputation'] += 5  # Reward for correct detection
            else:
                self.sensor_reputations[sensor_id]['reputation'] -= 20 # Penalize heavily for false alarms
            
            # Quarantine the sensor if its reputation drops too low.
            if self.sensor_reputations[sensor_id]['reputation'] < 50:
                self.sensor_reputations[sensor_id]['status'] = 'Quarantined'
                self.sensor_reputations[sensor_id]['reputation'] = 0 # Reputation cannot be negative
            
            # Create a transaction for this update and add it to a new block.
            tx = {'sensor_id': sensor_id, 'new_reputation': self.sensor_reputations[sensor_id]['reputation'], 'new_status': self.sensor_reputations[sensor_id]['status']}
            self.add_transaction_and_mine(tx)
    
    def get_sensor_status_df(self):
        """Returns the current state of all sensors as a DataFrame for display."""
        if not self.sensor_reputations:
            return pd.DataFrame(columns=['reputation', 'status']).rename_axis('Sensor ID')
        return pd.DataFrame.from_dict(self.sensor_reputations, orient='index')

# ==============================================================================
# 2. ADVANCED MACHINE LEARNING SETUP
# ==============================================================================
@st.cache_data
def load_and_prepare_data():
    file_path = './03_cleaned_data/Cleaned_IoT_Dataset.csv'
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    leaky_cols_to_remove = list(set(df.select_dtypes(include=['object']).columns.tolist() + [col for col in df.columns if 'modbus' in col] + ['type', 'thermostat_thermostat_status', 'weather_temperature']))
    X = df.drop(columns=['label'] + leaky_cols_to_remove)
    y = df['label']
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X); X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    return X, X_train, X_test, y_train, y_test, X.columns, scaler

@st.cache_resource
def train_models(X_train, y_train, X_train_normal):
    lgbm = lgb.LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
    X_train_reshaped = X_train_normal.values.reshape(X_train_normal.shape[0], 1, X_train_normal.shape[1])
    inputs = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])); encoded = LSTM(64, activation='relu')(inputs); decoded = RepeatVector(X_train_reshaped.shape[1])(encoded); decoded = LSTM(X_train_reshaped.shape[2], activation='sigmoid', return_sequences=True)(decoded)
    autoencoder = Model(inputs, decoded); autoencoder.compile(optimizer='adam', loss='mae'); autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    train_pred = autoencoder.predict(X_train_reshaped, verbose=0); train_mae_loss = np.mean(np.abs(train_pred - X_train_reshaped), axis=(1, 2)); threshold = np.max(train_mae_loss) * 1.5
    return lgbm, autoencoder, threshold

# ==============================================================================
# 3. STREAMLIT APPLICATION UI
# ==============================================================================
st.set_page_config(page_title="Industry 5.0 WSN Framework", layout="wide")
st.title("🛡️ Industry 5.0-Aligned Framework for WSN Resilience")

# --- Initialize Session State ---
if 'simulation_step' not in st.session_state: st.session_state.simulation_step = 0
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'history' not in st.session_state: st.session_state.history = {'lgbm': [], 'autoencoder': [], 'mcp': [], 'actual': []}
if 'blockchain' not in st.session_state: st.session_state.blockchain = Blockchain()
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'evaluation_report' not in st.session_state: st.session_state.evaluation_report = {}

# --- Load and train models ---
with st.spinner("Loading data and training advanced models..."):
    df_unscaled, X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data()
    X_train_normal = X_train[y_train == 0]
    lgbm_model, autoencoder_model, threshold = train_models(X_train, y_train, X_train_normal)
    sensor_prefixes = sorted(list(set([col.split('_')[0] for col in feature_names])))
st.success("Models trained and ready.")

# --- UI Layout ---
col1, col2 = st.columns([2, 1])
with col1:
    st.header("🔴 Digital Twin Simulation Environment")
    with st.sidebar:
        st.header("Digital Twin Controls")
        st.write("Configure the simulation environment to test the framework's resilience.")
        noise_level = st.slider("Noise Level (%)", 0, 50, 0, key='noise', help="Inject random Gaussian noise into sensor readings.")
        packet_drop_rate = st.slider("Channel Fault / Packet Drop (%)", 0, 50, 0, help="Simulate data loss by randomly dropping sensor readings.")
        faulty_sensor = st.selectbox("Designate a Consistently Faulty Sensor", options=["None"] + sensor_prefixes, help="This sensor will be blamed for all anomalies to test the quarantine system.")
    
    st.write("Use the controls to manage the simulation of the test dataset.")
    c1, c2, c3 = st.columns(3)
    if c1.button("▶️ Start / Resume"): st.session_state.is_running = True
    if c2.button("⏸️ Pause"): st.session_state.is_running = False
    if c3.button("🔁 Reset Simulation"):
        st.session_state.simulation_step = 0; st.session_state.is_running = False
        st.session_state.history = {'lgbm': [], 'autoencoder': [], 'mcp': [], 'actual': []}
        st.session_state.blockchain = Blockchain(); st.session_state.log_messages = []
        st.session_state.evaluation_report = {}

    st.subheader("📈 Live Performance Metrics"); metrics_placeholder = st.empty()
    st.subheader("📡 Network Status (Reputation Ledger)"); dashboard_placeholder = st.empty()
    st.subheader("📜 Live Event Log"); log_placeholder = st.empty()

with col2:
    st.header("🔬 Live Data Point Analysis")
    st.write("Manually create a data point and classify it in real-time.")
    live_inputs = {feature: st.slider(f"**{feature}**", float(df_unscaled[feature].min()), float(df_unscaled[feature].max()), float(df_unscaled[feature].mean())) for feature in feature_names}
    if st.button("Analyze Live Data Point"):
        live_df = pd.DataFrame([live_inputs]); live_df_scaled = scaler.transform(live_df)
        pred_lgbm_live = lgbm_model.predict(live_df_scaled)[0]
        live_reshaped = live_df_scaled.reshape(1, 1, live_df_scaled.shape[1])
        recon_error_live = np.mean(np.abs(autoencoder_model.predict(live_reshaped, verbose=0) - live_reshaped), axis=(1, 2))[0]
        pred_autoencoder_live = 1 if recon_error_live > threshold else 0
        pred_mcp_live = 1 if (pred_lgbm_live == 1 or pred_autoencoder_live == 1) else 0
        if pred_mcp_live == 1: st.error("#### 🚨 MCP VERDICT: ANOMALY DETECTED")
        else: st.success("#### ✅ MCP VERDICT: NORMAL")
        st.write(f"**LightGBM:** `{'Attack' if pred_lgbm_live == 1 else 'Normal'}`"); st.write(f"**LSTM Autoencoder:** `{'Attack' if pred_autoencoder_live == 1 else 'Normal'}`")

# ==============================================================================
# 4. SIMULATION & EVALUATION LOGIC
# ==============================================================================
if st.session_state.is_running and st.session_state.simulation_step < len(X_test):
    ENERGY_COST_PER_EVENT = 0.05; ENERGY_COST_FOR_ANOMALY = 0.2
    i = st.session_state.simulation_step
    X_test_faulty = X_test.copy()
    if noise_level > 0: X_test_faulty += np.random.normal(0, noise_level / 100, X_test_faulty.shape)
    if packet_drop_rate > 0:
        for col in X_test_faulty.columns: X_test_faulty.loc[X_test_faulty.sample(frac=packet_drop_rate / 100).index, col] = np.nan
        X_test_faulty = X_test_faulty.ffill().bfill()
    live_data_row = X_test_faulty.iloc[[i]]; actual_label = y_test.iloc[i]
    pred_lgbm = lgbm_model.predict(live_data_row)[0]
    live_data_reshaped = live_data_row.values.reshape(1, 1, live_data_row.shape[1])
    recon_error = np.mean(np.abs(autoencoder_model.predict(live_data_reshaped, verbose=0) - live_data_reshaped), axis=(1, 2))[0]
    pred_autoencoder = 1 if recon_error > threshold else 0
    pred_mcp = 1 if (pred_lgbm == 1 or pred_autoencoder == 1) else 0
    st.session_state.history['lgbm'].append(pred_lgbm); st.session_state.history['autoencoder'].append(pred_autoencoder); st.session_state.history['mcp'].append(pred_mcp); st.session_state.history['actual'].append(actual_label)
    log_msg = f"Event:{i+1},Actual:{'Attack' if actual_label==1 else 'Normal'},MCP Verdict:{'Attack' if pred_mcp==1 else 'Normal'}"
    logging.info(log_msg); st.session_state.log_messages.insert(0, log_msg)
    energy_without = st.session_state.evaluation_report.get('energy_without_framework', 0); energy_with = st.session_state.evaluation_report.get('energy_with_framework', 0)
    energy_without += ENERGY_COST_PER_EVENT
    if pred_mcp == 1: energy_without += ENERGY_COST_FOR_ANOMALY
    if pred_mcp == 1:
        is_true_positive = (actual_label == 1)
        if faulty_sensor != "None":
            involved_sensor = faulty_sensor
        else:
            most_extreme_feature = live_data_row.abs().idxmax(axis=1).iloc[0]
            involved_sensor = most_extreme_feature.split('_')[0]
        if st.session_state.blockchain.sensor_reputations.get(involved_sensor, {}).get('status', 'Active') == 'Active':
            energy_with += ENERGY_COST_PER_EVENT + ENERGY_COST_FOR_ANOMALY
            old_status = st.session_state.blockchain.sensor_reputations.get(involved_sensor, {}).get('status', 'Active')
            st.session_state.blockchain.update_reputation(involved_sensor, is_true_positive)
            new_status = st.session_state.blockchain.sensor_reputations[involved_sensor]['status']
            if new_status == 'Quarantined' and old_status == 'Active':
                if 'ttq' not in st.session_state.evaluation_report: st.session_state.evaluation_report['ttq'] = f"Quarantined after {i+1} events."
            st.session_state.log_messages.insert(0, f"Ledger: Updated reputation for sensor '{involved_sensor}'.")
        else:
            energy_with += ENERGY_COST_PER_EVENT
            st.session_state.log_messages.insert(0, f"Info: Sensor {involved_sensor} is quarantined. Ignoring alert.")
    else: energy_with += ENERGY_COST_PER_EVENT
    st.session_state.evaluation_report['energy_with_framework'] = energy_with; st.session_state.evaluation_report['energy_without_framework'] = energy_without
    with metrics_placeholder.container():
        c1, c2, c3 = st.columns(3)
        def safe_f1_score(y_true, y_pred): return f1_score(y_true, y_pred, zero_division=0.0)
        c1.metric("LGBM F1-Score", f"{safe_f1_score(st.session_state.history['actual'], st.session_state.history['lgbm']):.3f}")
        c2.metric("LSTM Autoencoder F1-Score", f"{safe_f1_score(st.session_state.history['actual'], st.session_state.history['autoencoder']):.3f}")
        c3.metric("MCP (Soft Vote) F1-Score", f"{safe_f1_score(st.session_state.history['actual'], st.session_state.history['mcp']):.3f}")
    dashboard_placeholder.dataframe(st.session_state.blockchain.get_sensor_status_df())
    log_placeholder.text_area("Log", "\n".join(st.session_state.log_messages), height=200)
    st.session_state.simulation_step += 1
    # *** THE DEFINITIVE FIX: Optimized sleep time for smooth UI ***
    time.sleep(0.001)
    st.rerun()
elif st.session_state.simulation_step > 0:
    with metrics_placeholder.container():
        c1, c2, c3 = st.columns(3)
        def safe_f1_score(y_true, y_pred): return f1_score(y_true, y_pred, zero_division=0.0)
        c1.metric("LGBM F1-Score", f"{safe_f1_score(st.session_state.history['actual'], st.session_state.history['lgbm']):.3f}")
        c2.metric("LSTM Autoencoder F1-Score", f"{safe_f1_score(st.session_state.history['actual'], st.session_state.history['autoencoder']):.3f}")
        c3.metric("MCP (Soft Vote) F1-Score", f"{safe_f1_score(st.session_state.history['actual'], st.session_state.history['mcp']):.3f}")
    dashboard_placeholder.dataframe(st.session_state.blockchain.get_sensor_status_df())
    log_placeholder.text_area("Log", "\n".join(st.session_state.log_messages), height=200)

    if not st.session_state.is_running and st.session_state.simulation_step >= len(X_test):
        st.success("✅ Live simulation complete!"); st.balloons()
        st.header("🔬 Final Evaluation Report")
        report = st.session_state.evaluation_report; energy_with = report.get('energy_with_framework', 0); energy_without = report.get('energy_without_framework', 0)
        r1, r2, r3 = st.columns(3)
        r1.metric("Time to Quarantine (TTQ)", value=report.get('ttq', "N/A"))
        try: energy_saved = ((energy_without - energy_with) / energy_without) * 100; r2.metric("Network Lifetime Improvement", f"{energy_saved:.2f} % Energy Saved")
        except ZeroDivisionError: r2.metric("Network Lifetime Improvement", "N/A")
        blockchain_size = sys.getsizeof(json.dumps(st.session_state.blockchain.chain)); r3.metric("Blockchain Storage Overhead", f"{blockchain_size} bytes")
        st.header("📊 Final Model Performance Comparison")
        final_metrics={"Model": ["LightGBM", "LSTM Autoencoder", "MCP (Soft Vote)"],"Accuracy": [accuracy_score(st.session_state.history['actual'], st.session_state.history['lgbm']), accuracy_score(st.session_state.history['actual'], st.session_state.history['autoencoder']), accuracy_score(st.session_state.history['actual'], st.session_state.history['mcp'])],"Precision": [precision_score(st.session_state.history['actual'], st.session_state.history['lgbm'], zero_division=0.0), precision_score(st.session_state.history['actual'], st.session_state.history['autoencoder'], zero_division=0.0), precision_score(st.session_state.history['actual'], st.session_state.history['mcp'], zero_division=0.0)],"Recall": [recall_score(st.session_state.history['actual'], st.session_state.history['lgbm'], zero_division=0.0), recall_score(st.session_state.history['actual'], st.session_state.history['autoencoder'], zero_division=0.0), recall_score(st.session_state.history['actual'], st.session_state.history['mcp'], zero_division=0.0)],"F1-Score": [f1_score(st.session_state.history['actual'], st.session_state.history['lgbm'], zero_division=0.0), f1_score(st.session_state.history['actual'], st.session_state.history['autoencoder'], zero_division=0.0), f1_score(st.session_state.history['actual'], st.session_state.history['mcp'], zero_division=0.0)]}
        results_df=pd.DataFrame(final_metrics).set_index("Model")
        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
        st.subheader("Final Confusion Matrix for MCP Model")
        cm=confusion_matrix(st.session_state.history['actual'], st.session_state.history['mcp']); fig, ax=plt.subplots(figsize=(6,4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack']); ax.set_xlabel('Predicted Label'); ax.set_ylabel('Actual Label'); st.pyplot(fig)