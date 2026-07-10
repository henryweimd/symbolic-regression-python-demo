# Symbolic Regression vs Linear Regression (Python Demo)

This is the Python equivalent of the Evolutionary AI dashboard, rebuilt using **Streamlit** and **PySR**.

## Features
* **PySR Symbolic Regression Engine**: Uses PySR (Julia-based) to discover complex, non-linear equations natively.
* **Dual Datasets**: Includes both the Parkinson's Clinical dataset and the Medical Insurance dataset.
* **Interactive UI**: Streamlit sidebar for tweaking evolutionary hyper-parameters (Population Size, Generations, Parsimony Penalty).
* **Plotly Visualizations**: Interactive scatter plots comparing standard Linear Regression to Evolutionary AI.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Deploy (Free)
1. Push this repository to GitHub.
2. Log into [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click "New App", select this repository, and hit Deploy. It will be live instantly.
