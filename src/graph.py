"""
graph.py - LangGraph Agentic ML Workflow for Breast Cancer Classification.

This module defines the workflow state and the nodes that represent the steps 
in our machine learning pipeline. The steps are orchestrated using LangGraph's 
StateGraph.
"""

from typing import TypedDict, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END
from sklearn.metrics import confusion_matrix

# Import the existing modular pipeline components
from src.data_loader import load_data
from src.preprocessing import split_data, scale_features
from src.model_training import train_models
from src.evaluation import evaluate_models
from src.explanation_agent import generate_pipeline_explanation
from src.report_generator import generate_markdown_report

# Define the shared state dictionary type.
class MLPipelineState(TypedDict):
    # Data storage
    df: Optional[pd.DataFrame]
    
    # Train/Test splits
    X_train: Optional[Any]
    X_test: Optional[Any]
    y_train: Optional[Any]
    y_test: Optional[Any]
    
    # Preprocessed/scaled features and the fitted scaler
    X_train_scaled: Optional[np.ndarray]
    X_test_scaled: Optional[np.ndarray]
    scaler: Optional[Any]
    
    # Dictionary of trained model names to their fitted estimator objects
    models: Optional[Dict[str, Any]]
    
    # Evaluation results DataFrame
    results_df: Optional[pd.DataFrame]
    
    # Best model details
    best_name: Optional[str]
    best_model: Optional[Any]
    
    # Textual summary, explanation, and report generation
    summary: Optional[str]
    explanation: Optional[str]
    report_path: Optional[str]
    report_content: Optional[str]


# --- Workflow Nodes ---

def load_data_node(state: MLPipelineState) -> Dict[str, Any]:
    """
    Node 1: Load the Breast Cancer dataset.
    """
    print("[Node] Loading dataset...")
    df = load_data()
    return {"df": df}


def preprocess_data_node(state: MLPipelineState) -> Dict[str, Any]:
    """
    Node 2: Preprocess the data.
    """
    print("[Node] Preprocessing data...")
    df = state["df"]
    if df is None:
        raise ValueError("DataFrame not found in state. Check load_data_node.")
        
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler
    }


def train_models_node(state: MLPipelineState) -> Dict[str, Any]:
    """
    Node 3: Train machine learning models.
    """
    print("[Node] Training models...")
    X_train_scaled = state["X_train_scaled"]
    y_train = state["y_train"]
    
    models = train_models(X_train_scaled, y_train)
    return {"models": models}


def evaluate_models_node(state: MLPipelineState) -> Dict[str, Any]:
    """
    Node 4: Evaluate the trained models.
    """
    print("[Node] Evaluating models...")
    models = state["models"]
    X_test_scaled = state["X_test_scaled"]
    y_test = state["y_test"]
    
    best_name, best_model, results_df = evaluate_models(models, X_test_scaled, y_test)
    
    return {
        "best_name": best_name,
        "best_model": best_model,
        "results_df": results_df
    }


def select_best_model_node(state: MLPipelineState) -> Dict[str, Any]:
    """
    Node 5: Decision node to finalize selection of the best model.
    """
    best_name = state["best_name"]
    results_df = state["results_df"]
    best_f1 = results_df.loc[results_df["Model"] == best_name, "F1-Score"].values[0]
    
    print(f"[Node] Final model selected: {best_name} (F1-Score: {best_f1})")
    return {"best_name": best_name}


def generate_summary_node(state: MLPipelineState) -> Dict[str, Any]:
    """
    Node 6: Generate text explanation, compile reports and summary.
    """
    df = state["df"]
    best_name = state["best_name"]
    results_df = state["results_df"]
    best_model = state["best_model"]
    X_test_scaled = state["X_test_scaled"]
    y_test = state["y_test"]

    # 1. Retrieve the metrics of the best model
    best_row = results_df.loc[results_df["Model"] == best_name].iloc[0]
    best_metrics = {
        "Accuracy": best_row["Accuracy"],
        "Precision": best_row["Precision"],
        "Recall": best_row["Recall"],
        "F1-Score": best_row["F1-Score"]
    }

    # 2. Get the confusion matrix
    y_pred = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    # 3. Generate explanation
    explanation = generate_pipeline_explanation(best_name, best_metrics, cm)

    # 4. Generate report
    class_counts = {
        "malignant": int((df["target"] == 0).sum()),
        "benign": int((df["target"] == 1).sum())
    }
    report_path, report_content = generate_markdown_report(
        df_shape=df.shape,
        class_counts=class_counts,
        results_df=results_df,
        best_name=best_name,
        best_cm=cm,
        explanation_text=explanation
    )

    summary_text = (
        f"Pipeline successfully ran using LangGraph orchestration.\n"
        f"The best classification model was chosen based on macro F1-score:\n"
        f"Winner: {best_name} with an F1-score of {best_metrics['F1-Score']:.4f}.\n"
        f"Comparison report saved in reports/model_comparison.csv."
    )
    
    print("[Node] Summary and explanation generated. Report saved.")
    return {
        "summary": summary_text,
        "explanation": explanation,
        "report_path": report_path,
        "report_content": report_content
    }


# --- Build LangGraph Workflow ---

def create_ml_pipeline_graph():
    """
    Create and compile the LangGraph workflow structure.
    """
    workflow = StateGraph(MLPipelineState)
    
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("preprocess_data", preprocess_data_node)
    workflow.add_node("train_models", train_models_node)
    workflow.add_node("evaluate_models", evaluate_models_node)
    workflow.add_node("select_best_model", select_best_model_node)
    workflow.add_node("generate_summary", generate_summary_node)
    
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "preprocess_data")
    workflow.add_edge("preprocess_data", "train_models")
    workflow.add_edge("train_models", "evaluate_models")
    workflow.add_edge("evaluate_models", "select_best_model")
    workflow.add_edge("select_best_model", "generate_summary")
    workflow.add_edge("generate_summary", END)
    
    app = workflow.compile()
    return app
