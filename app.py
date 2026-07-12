import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import threading
import time

import os

# Fix for Streamlit Cloud PermissionError, wiping, and Watchdog crashes
# Install Julia into the home directory (writable, persistent, and ignored by Streamlit's file watcher)
os.environ["JULIA_DEPOT_PATH"] = os.path.expanduser("~/.julia_depot")
os.environ["PYTHON_JULIAPKG_PROJECT"] = os.path.expanduser("~/.juliapkg")
os.environ["JULIA_NUM_THREADS"] = "1"  # Prevent OOM memory crash during compilation
os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"  # Disable aggressive precompilation to prevent boot timeouts and OOM. Defers to JIT.

import plotly.express as px

st.set_page_config(layout="wide", page_title="Symbolic Regression vs Linear Regression")

@st.cache_resource
def get_compilation_status():
    return {"done": False, "start_time": None, "error": None}

compilation_status = get_compilation_status()

@st.cache_resource
def start_julia_background_compilation():
    def _compile():
        compilation_status["start_time"] = time.time()
        try:
            from pysr import PySRRegressor
            compilation_status["done"] = True
        except Exception as e:
            compilation_status["error"] = str(e)
            compilation_status["done"] = True
            
    t = threading.Thread(target=_compile)
    t.start()
    return True

start_julia_background_compilation()

st.title("Symbolic Regression vs Linear Regression")
st.markdown("This dashboard demonstrates how **Symbolic Regression** (via Evolutionary AI) discovers complex non-linear mathematical equations from data, outperforming traditional Linear Regression on datasets with hidden interactions or noise.")

# --- Sidebar ---
st.sidebar.header("1. Choose Dataset")
dataset_choice = st.sidebar.radio(
    "Select Dataset",
    ["Parkinson's Disease (Clinical Data)", "Medical Insurance (Actuarial Data)"]
)

if dataset_choice == "Parkinson's Disease (Clinical Data)":
    st.sidebar.markdown("**Parkinson's Data:** Noisy clinical measurements of vocal cord degradation. Target is `motor_UPDRS` (severity score).")
else:
    st.sidebar.markdown("**Insurance Data:** Clean actuarial data. The target is `charges`. Contains a hidden `bmi * smoker` interaction that Linear Regression fails on.")

st.sidebar.header("2. Evolutionary AI Parameters")
populations = st.sidebar.slider("Population Size (Equations per generation)", min_value=10, max_value=200, value=50, step=10)
niterations = st.sidebar.slider("Generations (Evolution cycles)", min_value=10, max_value=200, value=30, step=10)
parsimony = st.sidebar.slider("Complexity Penalty", min_value=0.0, max_value=0.1, value=0.01, step=0.01)

run_button = st.sidebar.button("Run AI Evolution", type="primary")

st.sidebar.markdown("---")

@st.fragment(run_every=1)
def show_compilation_status():
    if not compilation_status["done"] and compilation_status["start_time"] is not None:
        elapsed = int(time.time() - compilation_status["start_time"])
        estimated = 180  # ~3 minutes based on logs
        progress = min(elapsed / estimated, 0.99)  # Hold at 99% until truly done
        st.sidebar.progress(progress, text=f"⚙️ Precompiling Julia engine... ({elapsed}s / ~180s)")
    elif compilation_status["done"] and not compilation_status["error"]:
        st.sidebar.success("✅ Julia math engine ready!")
    elif compilation_status["error"]:
        st.sidebar.error(f"⚠️ Compilation failed: {compilation_status['error']}")

show_compilation_status()

# --- Data Loading ---
@st.cache_data
def load_data(choice):
    if choice == "Parkinson's Disease (Clinical Data)":
        df = pd.read_csv("parkinsons_updrs.data")
        # Rename for easier parsing
        df = df.rename(columns={"Jitter(Abs)": "JitterAbs"})
        # Sample to 300 rows for speed in demo
        df = df.sample(n=min(300, len(df)), random_state=42)
        X = df[['age', 'JitterAbs', 'Shimmer', 'HNR', 'PPE']]
        y = df['motor_UPDRS']
        return df, X, y, "motor_UPDRS"
    else:
        df = pd.read_csv("insurance.csv")
        df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
        df = df.sample(n=min(300, len(df)), random_state=42)
        X = df[['age', 'bmi', 'smoker']]
        y = df['charges']
        return df, X, y, "charges"

df, X, y, target_col = load_data(dataset_choice)

st.write("### Data Preview")
st.dataframe(df.head())

# --- Model Execution ---
if run_button:
    col1, col2 = st.columns(2)
    
    # 1. Linear Regression
    with col1:
        st.subheader("📊 Traditional Linear Regression")
        lm = LinearRegression()
        lm.fit(X, y)
        lm_preds = lm.predict(X)
        lm_r2 = r2_score(y, lm_preds)
        
        # Build LM Equation String
        intercept = lm.intercept_
        coefs = lm.coef_
        eq_terms = [f"{coef:.2f} * {col}" for coef, col in zip(coefs, X.columns)]
        lm_eq = f"{target_col} = {intercept:.2f} + " + " + ".join(eq_terms)
        
        st.info(f"**R-Squared:** {lm_r2:.4f}")
        st.markdown("**Discovered Equation:**")
        st.code(lm_eq, language="python")
        
        # Plot
        fig_lm = px.scatter(x=y, y=lm_preds, labels={'x': f'Actual {target_col}', 'y': f'Predicted {target_col}'}, title="Linear Regression: Actual vs Predicted")
        fig_lm.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_lm, use_container_width=True)
        
    # 2. Symbolic Regression (PySR)
    with col2:
        st.subheader("🧬 Evolutionary AI (Symbolic Regression)")
        with st.spinner("Breeding equations... this may take a moment."):
            from pysr import PySRRegressor
            # Configure PySR
            model = PySRRegressor(
                niterations=niterations,
                populations=populations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp"],
                parsimony_penalty=parsimony,
                verbosity=0,
                random_state=42,
                deterministic=True,
                procs=1,
                multithreading=False # safer for cloud hosting environments
            )
            model.fit(X, y)
            
            pysr_preds = model.predict(X)
            pysr_r2 = r2_score(y, pysr_preds)
            best_eq = model.sympy()
            
        st.success(f"**R-Squared:** {pysr_r2:.4f}")
        st.markdown("**Discovered Equation:**")
        st.code(f"{target_col} = {best_eq}", language="python")
        
        # Plot
        fig_sr = px.scatter(x=y, y=pysr_preds, labels={'x': f'Actual {target_col}', 'y': f'Predicted {target_col}'}, title="Symbolic Regression: Actual vs Predicted")
        fig_sr.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_sr, use_container_width=True)
