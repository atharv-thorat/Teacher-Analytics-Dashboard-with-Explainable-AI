"""
app.py - Student Risk Performance Analytics with Explainable AI
Domain-Specific Risk Prediction | RF+SVM Ensemble | SHAP + LIME
UPDATED: Better target selection and composite risk scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import StudentRiskModel

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CONSTANTS
# ============================================
REQUIRED_FEATURES = [
    "attendance_percentage",
    "midterm_marks",
    "final_marks",
    "internal_assessment",
    "assignment_submission_rate"
]

# ============================================
# SESSION STATE
# ============================================
if "model" not in st.session_state:
    st.session_state.model = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = None

# ============================================
# HELPER FUNCTIONS
# ============================================
def create_composite_target(data):
    """
    Create a composite academic performance score that properly weights
    academic performance (exams) more heavily than attendance
    """
    # Normalize all features to 0-100 scale
    df = data.copy()
    
    # Academic performance (70% weight) - what really matters
    academic_score = (
        0.35 * df["midterm_marks"] +           # 35% - midterm performance
        0.35 * df["final_marks"] +             # 35% - final performance  
        0.15 * df["internal_assessment"] +     # 15% - internal assessment
        0.15 * df["assignment_submission_rate"] # 15% - assignments
    )
    
    # Behavioral factors (30% weight) - supporting indicators
    behavioral_score = (
        1.0 * df["attendance_percentage"]      # 30% - attendance
    )
    
    # Composite score: 70% academic, 30% behavioral
    composite = 0.70 * academic_score + 0.30 * behavioral_score
    
    return composite

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        "Low Risk": "#4CAF50",
        "Medium Risk": "#FFC107", 
        "High Risk": "#F44336"
    }
    return colors.get(risk_level, "#999")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    st.markdown("---")
    st.markdown("""
    ### 📋 Quick Guide
    1. **Upload** student data CSV
    2. **Configure** target metric  
    3. **Train** AI model
    4. **Explore** predictions & explanations
    
    ### 🎯 What We Predict
    Student academic risk based on:
    - 📚 **Exam Performance** (midterm, final)
    - 📝 **Assessments** (internal, assignments)
    - 📅 **Attendance** (class participation)
    
    ### 💡 Key Features
    - ✅ Composite scoring (academics + behavior)
    - 🔍 AI explanations (SHAP + LIME)
    - 📊 Interactive visualizations
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Academic performance (exams) is weighted more heavily than attendance in risk calculation")

# ============================================
# MAIN APP
# ============================================
def main():
    st.title("🎓 Student Performance Analytics")
    st.markdown("### AI-Powered Academic Risk Prediction System")

    # --------------------------------------------
    # STEP 1: UPLOAD
    # --------------------------------------------
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file to begin analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎯 What This Tool Does
            - Predicts student academic risk levels
            - Identifies at-risk students early
            - Explains AI predictions with SHAP & LIME
            - Provides actionable insights for educators
            """)
        
        with col2:
            st.markdown("""
            #### 📊 Required Data Columns
            - `attendance_percentage`
            - `midterm_marks`
            - `final_marks`
            - `internal_assessment`
            - `assignment_submission_rate`
            """)
        
        st.markdown("---")
        st.warning("⚠️ **Note**: The model prioritizes academic performance (exams) over attendance when assessing risk")
        st.stop()

    # Load data
    data = pd.read_csv(uploaded_file)
    st.session_state.data = data

    st.success(f"✅ Dataset loaded: **{len(data)} students**, **{len(data.columns)} features**")

    with st.expander("📋 View Dataset", expanded=False):
        st.dataframe(data.head(20), use_container_width=True)
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(data))
        col2.metric("Features", len(data.columns))
        col3.metric("Complete Records", data.dropna().shape[0])

    # --------------------------------------------
    # STEP 2: SCHEMA VALIDATION
    # --------------------------------------------
    missing = [f for f in REQUIRED_FEATURES if f not in data.columns]
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        st.info("Please ensure your CSV contains all required columns")
        st.stop()

    # --------------------------------------------
    # STEP 3: TARGET SELECTION
    # --------------------------------------------
    st.markdown("---")
    st.header("⚙️ Model Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Risk Assessment Target")
        
        target_option = st.radio(
            "Select what to predict risk from:",
            [
                "📊 Composite Score (Recommended - Balanced academic + behavioral)",
                "📚 Academic Performance Only (Exams + assessments)",
                "📝 Final Exam Performance",
                "📈 Overall GPA/Marks Average"
            ],
            help="Composite score weighs academics 70% and attendance 30%"
        )
        
        # Determine target based on selection
        if "Composite" in target_option:
            st.info("✨ **Composite scoring**: 70% academics (midterm, final, assessments) + 30% attendance")
            data["target_score"] = create_composite_target(data)
            target_col = "target_score"
            
        elif "Academic Performance Only" in target_option:
            st.info("📚 Using pure academic metrics (excluding attendance)")
            data["target_score"] = (
                0.40 * data["midterm_marks"] +
                0.40 * data["final_marks"] +
                0.20 * data["internal_assessment"]
            )
            target_col = "target_score"
            
        elif "Final Exam" in target_option:
            st.info("📝 Predicting risk based on final exam performance")
            target_col = "final_marks"
            
        else:  # Overall average
            st.info("📈 Using overall marks average")
            data["target_score"] = data[["midterm_marks", "final_marks", "internal_assessment"]].mean(axis=1)
            target_col = "target_score"

    with col2:
        st.subheader("📊 Input Features")
        st.markdown("**Model uses:**")
        for f in REQUIRED_FEATURES:
            st.markdown(f"✓ `{f}`")

    # Show target distribution
    with st.expander("📈 Target Score Distribution", expanded=False):
        if target_col in data.columns:
            fig = px.histogram(
                data, 
                x=target_col,
                nbins=30,
                title=f"Distribution of {target_col}",
                labels={target_col: "Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Score", f"{data[target_col].mean():.2f}")
            col2.metric("Median Score", f"{data[target_col].median():.2f}")
            col3.metric("Std Dev", f"{data[target_col].std():.2f}")

    # --------------------------------------------
    # STEP 4: TRAIN MODEL
    # --------------------------------------------
    st.markdown("---")
    st.header("🚀 Train Model")

    col1, col2 = st.columns([3, 1])
    
    with col1:
        train_button = st.button("🎯 Train AI Model", use_container_width=True, type="primary")
    
    with col2:
        if st.session_state.trained:
            st.success("✓ Trained")

    if train_button:
        with st.spinner("🔄 Training ensemble model (RF + SVM)..."):
            model = StudentRiskModel()
            
            # Train with proper weighting
            metrics = model.train(
                data=data,
                target_col=target_col,
                feature_cols=REQUIRED_FEATURES,
                feature_weights=None  # Let model determine optimal weights
            )

            st.session_state.model = model
            st.session_state.trained = True

            st.success("✅ Model trained successfully!")

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['precision']:.2%}")
            col3.metric("Recall", f"{metrics['recall']:.2%}")
            col4.metric("AUC Score", f"{metrics['auc']:.2f}")

    # --------------------------------------------
    # STOP IF NOT TRAINED
    # --------------------------------------------
    if not st.session_state.trained or st.session_state.model is None:
        st.info("📌 Train the model above to view predictions and explanations")
        st.stop()

    model = st.session_state.model

    # --------------------------------------------
    # STEP 5: PREDICTIONS & RESULTS
    # --------------------------------------------
    st.markdown("---")
    st.header("📈 Risk Assessment Results")

    # Generate predictions
    with st.spinner("Generating predictions..."):
        probs = model.predict(data)
        results = data.copy()
        results["Risk_Score"] = probs
        results["Risk_Level"] = model.get_risk_categories(probs)
        results["Confidence"] = model.get_confidence(probs)
        st.session_state.results = results

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Students", len(results))
    col2.metric("🔴 High Risk", (results["Risk_Level"] == "High Risk").sum())
    col3.metric("🟡 Medium Risk", (results["Risk_Level"] == "Medium Risk").sum())
    col4.metric("🟢 Low Risk", (results["Risk_Level"] == "Low Risk").sum())

    # --------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------
    st.subheader("📊 Risk Distribution Visualizations")
    
    col1, col2 = st.columns(2)

    with col1:
        # Pie chart
        risk_counts = results["Risk_Level"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color=risk_counts.index,
            color_discrete_map={
                "Low Risk": "#4CAF50",
                "Medium Risk": "#FFC107",
                "High Risk": "#F44336"
            },
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk score histogram
        fig = px.histogram(
            results,
            x="Risk_Score",
            nbins=40,
            title="Risk Score Distribution",
            color="Risk_Level",
            color_discrete_map={
                "Low Risk": "#4CAF50",
                "Medium Risk": "#FFC107",
                "High Risk": "#F44336"
            },
            labels={"Risk_Score": "Risk Probability"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("🎯 Feature Importance Analysis")
    
    importance = model.get_feature_importance()
    imp_df = pd.DataFrame([
        {"Feature": k, "Importance": v}
        for k, v in importance.items()
    ]).sort_values("Importance", ascending=True)
    
    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Which factors matter most in predicting risk?",
        color="Importance",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------
    st.subheader("📋 Detailed Student Risk Table")
    
    # Add risk level filter
    risk_filter = st.multiselect(
        "Filter by risk level:",
        ["Low Risk", "Medium Risk", "High Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"]
    )
    
    filtered_results = results[results["Risk_Level"].isin(risk_filter)]
    
    # Display table with color coding
    st.dataframe(
        filtered_results.sort_values("Risk_Score", ascending=False).style.background_gradient(
            subset=["Risk_Score"],
            cmap="RdYlGn_r"
        ),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="student_risk_predictions.csv",
        mime="text/csv"
    )

    # --------------------------------------------
    # INDIVIDUAL STUDENT ANALYSIS
    # --------------------------------------------
    st.markdown("---")
    st.header("🔍 Individual Student Deep Dive")

    # Student selector with better formatting
    col1, col2 = st.columns([3, 1])
    
    with col1:
        student_idx = st.selectbox(
            "Select a student to analyze:",
            options=results.index,
            format_func=lambda i: f"Student #{i} — {results.loc[i, 'Risk_Level']} ({results.loc[i, 'Risk_Score']:.2%})"
        )
    
    with col2:
        if st.button("🔄 Random Student"):
            import random
            student_idx = random.choice(results.index.tolist())
            st.rerun()

    # Student overview
    st.markdown(f"### Student #{student_idx} Profile")
    
    risk_level = results.loc[student_idx, "Risk_Level"]
    risk_score = results.loc[student_idx, "Risk_Score"]
    confidence = results.loc[student_idx, "Confidence"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Risk Level",
            risk_level,
            delta="Attention Needed" if risk_level == "High Risk" else "On Track"
        )
    
    with col2:
        st.metric(
            "Risk Score", 
            f"{risk_score:.1%}",
            delta=f"{abs(risk_score - 0.5):.1%} from threshold"
        )
    
    with col3:
        st.metric("Model Confidence", confidence)

    # Student feature values
    with st.expander("📊 View Student's Raw Features", expanded=False):
        student_data = results.loc[student_idx, REQUIRED_FEATURES]
        
        fig = go.Figure(data=[
            go.Bar(
                x=student_data.values,
                y=student_data.index,
                orientation='h',
                marker_color='steelblue'
            )
        ])
        fig.update_layout(
            title="Student's Feature Values",
            xaxis_title="Value",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------
    st.markdown("---")
    st.subheader("🧠 AI Explanation: What's Driving This Student's Risk?")
    st.markdown("**SHAP Analysis** - Shows which factors increase or decrease risk")

    with st.spinner("Generating SHAP explanation..."):
        shap_data = model.explain_shap(data, student_idx)
    
    # Build dataframe with correct key names from explanations.py
    shap_rows = []
    for exp in shap_data:
        shap_rows.append({
            "Feature": exp["feature"],
            "Impact": exp["shap_value"],
            "Abs_Impact": abs(exp["shap_value"]),
            "Direction": "⬆️ Increases Risk" if exp["shap_value"] > 0 else "⬇️ Reduces Risk",
            "Value": exp["feature_value"],  # Changed from "value" to "feature_value"
            "Percentile": exp["percentile"]
        })
    
    shap_df = pd.DataFrame(shap_rows).sort_values("Abs_Impact", ascending=False)

    # SHAP waterfall chart
    fig = px.bar(
        shap_df,
        y="Feature",
        x="Impact",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "⬆️ Increases Risk": "#F44336",
            "⬇️ Reduces Risk": "#4CAF50"
        },
        title="Feature Impact on Risk Prediction (SHAP Values)",
        labels={"Impact": "SHAP Value (impact on prediction)"}
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Feature contribution pie chart
    st.subheader("📊 Relative Feature Contributions")
    
    pie_df = shap_df.copy()
    pie_df["Contribution"] = (pie_df["Abs_Impact"] / pie_df["Abs_Impact"].sum()) * 100
    
    fig = px.pie(
        pie_df,
        values="Contribution",
        names="Feature",
        title="Which features matter most for this prediction?",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    # Detailed SHAP table
    with st.expander("📋 Detailed SHAP Analysis", expanded=False):
        st.dataframe(
            shap_df[["Feature", "Value", "Impact", "Percentile", "Direction"]].style.background_gradient(
                subset=["Impact"],
                cmap="RdYlGn_r"
            ),
            use_container_width=True
        )

    # --------------------------------------------
    # LIME EXPLANATION
    # --------------------------------------------
    st.markdown("---")
    st.subheader("🔬 Alternative Explanation: LIME Analysis")
    st.markdown("**LIME** provides a different perspective on the prediction")

    with st.spinner("Generating LIME explanation..."):
        lime_explanations = model.explain_lime(data, student_idx)

    for exp in lime_explanations:
        if "Increases risk" in exp["message"]:
            st.error(f"🔴 {exp['message']}")
        else:
            st.success(f"🟢 {exp['message']}")

    # --------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------
    st.markdown("---")
    st.subheader("💡 Actionable Recommendations")
    
    if risk_level == "High Risk":
        st.error("⚠️ **High Priority Intervention Needed**")
        st.markdown("""
        **Suggested Actions:**
        - 🎯 Schedule immediate one-on-one meeting
        - 📚 Develop personalized study plan
        - 👥 Connect with academic advisor
        - 📊 Monitor progress weekly
        """)
    elif risk_level == "Medium Risk":
        st.warning("⚡ **Early Intervention Recommended**")
        st.markdown("""
        **Suggested Actions:**
        - 📝 Provide additional practice materials
        - 👥 Recommend peer tutoring
        - 📅 Check-in bi-weekly
        - 🎓 Offer study skills workshop
        """)
    else:
        st.success("✅ **Student On Track**")
        st.markdown("""
        **Maintain Success:**
        - 🌟 Recognize achievements
        - 💪 Encourage continued effort
        - 📈 Monitor for any changes
        """)


if __name__ == "__main__":
    main()