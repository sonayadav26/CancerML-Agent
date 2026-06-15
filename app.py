"""
app.py - Streamlit frontend for the CancerML Agent pipeline using LangGraph.

Run from the terminal with:
    streamlit run app.py

This file imports the LangGraph workflow from src/graph.py and runs the
orchestrated ML agent pipeline, presenting the results in a Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import our LangGraph workflow creator
from src.graph import create_ml_pipeline_graph


# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="CancerML Agent",
    page_icon="🧬",
    layout="wide",
)

# ── Custom CSS for a cleaner look ─────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2.4rem; font-weight: 700; color: #1a73e8; }
    .sub-title   { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 0.5rem;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .disclaimer  {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 1rem; border-radius: 6px; margin-top: 1.5rem;
        color: #856404; font-size: 0.9rem;
    }
    .node-header {
        font-size: 1.25rem; font-weight: 600; color: #4b5563; margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🧬 CancerML Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">'
    'Agentic ML Assistant for Breast Cancer Classification &mdash; '
    'Orchestrated by LangGraph'
    '</p>',
    unsafe_allow_html=True,
)

# ── Project objective ─────────────────────────────────────────────
st.markdown("""
**Objective:**  
This app runs a modular machine-learning pipeline orchestrated using a **LangGraph state workflow**.
The workflow executes sequential nodes to load data, preprocess, train models, evaluate, and select the best model.

The three models trained and evaluated are:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
""")

st.divider()

# ── Run button ────────────────────────────────────────────────────
run_clicked = st.button("🚀 Run ML Pipeline (LangGraph Workflow)", type="primary", width="stretch")

if run_clicked:
    # Compile the LangGraph ML Pipeline
    pipeline_graph = create_ml_pipeline_graph()
    
    # Define initial state
    initial_state = {
        "df": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "X_train_scaled": None,
        "X_test_scaled": None,
        "scaler": None,
        "models": None,
        "results_df": None,
        "best_name": None,
        "best_model": None,
        "summary": None,
        "explanation": None,
        "report_path": None,
        "report_content": None
    }
    
    # Run the graph and display progress node-by-node
    st.subheader("⛓️ Workflow Node Execution Logs")
    
    # To capture state changes per node, we can stream updates from the graph.
    # Each update contains the state dictionary at the end of the node's execution.
    with st.status("Running LangGraph workflow nodes...", expanded=True) as status:
        state = initial_state
        for event in pipeline_graph.stream(initial_state):
            # The event dictionary has node names as keys and the state updates as values.
            for node_name, state_update in event.items():
                st.write(f"✔️ Node **{node_name}** executed successfully.")
                # Merge update into our local state tracking (safely handles None or non-dictionary updates)
                if state_update is not None and isinstance(state_update, dict):
                    state.update(state_update)
        status.update(label="LangGraph Workflow complete!", state="complete", expanded=False)
        
    st.success("Workflow executed successfully!")
    
    # Extract data from the final state of the graph
    df = state.get("df")
    X_train = state.get("X_train")
    X_test = state.get("X_test")
    models = state.get("models")
    results_df = state.get("results_df")
    best_name = state.get("best_name")
    summary = state.get("summary")
    explanation = state.get("explanation")
    report_content = state.get("report_content")
    X_test_scaled = state.get("X_test_scaled")
    y_test = state.get("y_test")

    if df is not None:
        # -- Dataset summary --
        st.header("📋 Dataset Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card"><h2>{df.shape[0]}</h2><p>Samples</p></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card"><h2>{df.shape[1] - 1}</h2><p>Features</p></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card"><h2>{(df["target"] == 0).sum()}</h2>'
                f'<p>Malignant (0)</p></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="metric-card"><h2>{(df["target"] == 1).sum()}</h2>'
                f'<p>Benign (1)</p></div>',
                unsafe_allow_html=True,
            )

        st.subheader("First 10 rows")
        st.dataframe(df.head(10), width="stretch")

        # -- Missing values report --
        st.subheader("🔍 Missing Values Report")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.info("No missing values found in the dataset. The data is clean!")
        else:
            missing_df = missing[missing > 0].reset_index()
            missing_df.columns = ["Feature", "Missing Count"]
            st.dataframe(missing_df, width="stretch")

    if results_df is not None:
        st.divider()

        # -- Model comparison table --
        st.header("📊 Model Comparison")
        st.dataframe(
            results_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                color="#d4edda",
            ),
            width="stretch",
        )

        # -- Best model announcement --
        best_f1 = results_df.loc[results_df["Model"] == best_name, "F1-Score"].values[0]
        st.success(f"🏆 **Best Model: {best_name}** with F1-Score = **{best_f1}**")

    if models is not None and X_test_scaled is not None and y_test is not None:
        st.divider()

        # -- Confusion matrices (one per model) --
        st.header("🔢 Confusion Matrices")
        cm_cols = st.columns(len(models))

        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(4, 3.5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Malignant", "Benign"],
            )
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(name, fontsize=11, fontweight="bold")
            fig.tight_layout()

            with cm_cols[idx]:
                st.pyplot(fig)
            plt.close(fig)  # free memory

    if explanation is not None:
        st.divider()
        st.header("🧠 Results Explanation")
        st.markdown(explanation)

    if report_content is not None:
        st.divider()
        st.header("💾 Download Full Markdown Report")
        st.download_button(
            label="📥 Download Pipeline Report (.md)",
            data=report_content,
            file_name="pipeline_run_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    if summary is not None:
        st.divider()
        st.header("📝 Pipeline Run Summary")
        st.info(summary)

    st.divider()

    # ── Educational disclaimer ────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        "<strong>⚠️ Educational Disclaimer:</strong> "
        "This application is a <em>learning project</em> built for educational "
        "purposes only. It is <strong>not</strong> a medical diagnosis tool. "
        "Never use predictions from this app to make real healthcare decisions. "
        "Always consult a qualified medical professional for any health concerns."
        "</div>",
        unsafe_allow_html=True,
    )

else:
    # ── Landing state (before the button is pressed) ──────────────
    st.info("👆 Click the **Run ML Pipeline** button above to start the agentic workflow.")

    st.markdown(
        '<div class="disclaimer">'
        "<strong>⚠️ Educational Disclaimer:</strong> "
        "This application is a <em>learning project</em> built for educational "
        "purposes only. It is <strong>not</strong> a medical diagnosis tool. "
        "Never use predictions from this app to make real healthcare decisions. "
        "Always consult a qualified medical professional for any health concerns."
        "</div>",
        unsafe_allow_html=True,
    )
