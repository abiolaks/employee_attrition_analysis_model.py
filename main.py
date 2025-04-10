import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from model import AttritionModel
from bias_mitigator import BiasMitigator
from llm_analyst import LLMAnalyst
import warnings

warnings.filterwarnings("ignore")


def main():
    st.set_page_config(
        page_title="Employee Attrition Analysis", page_icon="📊", layout="wide"
    )

    # Initialize components
    data_processor = DataProcessor("employee_attrition_data.csv")
    bias_mitigator = BiasMitigator()
    llm_analyst = LLMAnalyst()  # No API key by default

    # Load data
    data_processor.load_and_preprocess_data()
    X_train, X_test, y_train, y_test, preprocessor = (
        data_processor.get_preprocessed_data()
    )
    X_train_balanced, y_train_balanced = bias_mitigator.balance_data(X_train, y_train)

    # Sidebar controls
    st.sidebar.header("Configuration")
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "Gradient Boosting", "SVM"],
        index=0,
    )

    analyze_fairness = st.sidebar.checkbox("Analyze Model Fairness", value=True)
    sensitive_features = st.sidebar.multiselect(
        "Sensitive Attributes",
        ["Gender", "Marital Status", "Education Status"],
        default=["Gender"],
    )

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Analyzing data..."):
            # Train models
            attrition_model = AttritionModel(preprocessor)
            attrition_model.train_models(X_train_balanced, y_train_balanced)

            # Evaluate
            evaluation_results = attrition_model.evaluate_models(X_test, y_test)
            feature_importance = attrition_model.get_feature_importance(selected_model)

            # Fairness analysis
            fairness_report = None
            if analyze_fairness:
                best_model, _ = attrition_model.get_best_model()
                fairness_report = bias_mitigator.check_fairness(
                    best_model, X_test, y_test, sensitive_features
                )

            # LLM insights
            llm_insights = llm_analyst.generate_insights(
                data_processor.data,
                y_train.mean(),
                feature_importance.head(5)
                if feature_importance is not None
                else pd.DataFrame(),
                fairness_report or {},
            )

            # Display results
            display_results(
                data_processor.data,
                evaluation_results,
                feature_importance,
                fairness_report,
                llm_insights,
            )


def display_results(
    data, evaluation_results, feature_importance, fairness_report, llm_insights
):
    """Render all results in Streamlit"""
    st.title("Employee Attrition Analysis")

    # Key metrics
    st.header("📈 Key Metrics")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Attrition Rate", f"{data['Attrition'].mean():.1%}")
    with cols[1]:
        st.metric(
            "Best Model Accuracy",
            f"{max([res['accuracy'] for res in evaluation_results.values()]):.1%}",
        )

    # Model comparison
    st.header("🤖 Model Performance")
    st.dataframe(pd.DataFrame(evaluation_results).T.style.format("{:.2%}"))

    # Feature importance
    if feature_importance is not None:
        st.header("🔍 Top Factors")
        fig = px.bar(
            feature_importance.head(10), x="importance", y="feature", orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fairness analysis
    if fairness_report:
        st.header("⚖️ Fairness Analysis")
        for feature, metrics in fairness_report.items():
            st.subheader(feature)
            st.dataframe(pd.DataFrame(metrics).T)

    # LLM insights
    st.header("💡 AI Insights")
    st.write(llm_insights)

    # Data exploration
    st.header("🔎 Data Exploration")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Department")
        fig = px.bar(
            data.groupby("Role")["Attrition"].mean().sort_values(), orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Salary Distribution")
        fig = px.box(data, x="Attrition", y="Salary")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
    st.balloons()
