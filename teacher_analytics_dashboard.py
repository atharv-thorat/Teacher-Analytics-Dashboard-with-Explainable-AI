import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
import plotly.express as px

st.set_page_config(page_title="Teacher Dashboard - Student Performance", layout="wide")
st.title("🎓 Teacher-Friendly Student Performance Dashboard")
st.markdown("""
Upload a UCI Student Performance CSV (`Math` or `Portuguese`) and explore **Pass/Fail predictions**, top at-risk students, and SHAP explanations interactively.
""")

# ---------------------------
# 1. Upload Dataset
# ---------------------------
st.sidebar.header("1️⃣ Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # 2. Interactive Filtering
    # ---------------------------
    st.sidebar.header("2️⃣ Filter Data")
    filter_columns = st.sidebar.multiselect(
        "Select features to filter",
        df.columns.tolist(),
        default=['studytime','absences','sex','school']
    )
    filtered_df = df.copy()

    for col in filter_columns:
        if df[col].dtype == 'object':
            selected_vals = st.sidebar.multiselect(f"{col} filter", df[col].unique(), default=df[col].unique())
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
        else:
            min_val, max_val = int(df[col].min()), int(df[col].max())
            selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(filtered_df[col]>=selected_range[0]) & (filtered_df[col]<=selected_range[1])]

    st.write(f"Filtered dataset: {filtered_df.shape[0]} students")

    # ---------------------------
    # 3. Define Target
    # ---------------------------
    st.sidebar.header("3️⃣ Pass Threshold")
    pass_threshold = st.sidebar.slider("Pass threshold for G3", 0, 20, 10)
    filtered_df['pass_fail'] = (filtered_df['G3'] >= pass_threshold).astype(int)

    # ---------------------------
    # 4. Preprocessing
    # ---------------------------
    feature_cols = [c for c in filtered_df.columns if c not in ['G3','pass_fail']]
    X = filtered_df[feature_cols]
    y = filtered_df['pass_fail']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64','float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # ---------------------------
    # 5. Train Model
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X_train, y_train)

    st.subheader("Model Performance")
    st.write(f"✅ Train Accuracy: {model.score(X_train, y_train):.2f}")
    st.write(f"✅ Test Accuracy: {model.score(X_test, y_test):.2f}")

    # ---------------------------
    # 6. SHAP Global Feature Importance
    # ---------------------------
    st.subheader("Global Feature Importance")
    explainer = shap.TreeExplainer(model.named_steps['rf'])
    X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
    shap_values = explainer.shap_values(X_test_transformed)

    num_features = numerical_features
    cat_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = list(num_features) + list(cat_features)

    shap_array = shap_values[1]
    if shap_array.shape[1] > len(all_features):
        shap_array = shap_array[:, :len(all_features)]
    elif shap_array.shape[1] < len(all_features):
        shap_array = np.hstack([shap_array, np.zeros((shap_array.shape[0], len(all_features) - shap_array.shape[1]))])

    shap_summary_df = pd.DataFrame({
        "Feature": all_features,
        "Mean(|SHAP value|)": np.abs(shap_array).mean(axis=0)
    }).sort_values("Mean(|SHAP value|)", ascending=True)

    fig_global = px.bar(
        shap_summary_df,
        x="Mean(|SHAP value|)",
        y="Feature",
        orientation="h",
        title="Global Feature Importance (SHAP)"
    )
    st.plotly_chart(fig_global)

    # ---------------------------
    # 7. Individual Student Explanation
    # ---------------------------
    st.subheader("Explain Prediction for a Single Student")
    student_idx = st.slider("Select test sample index", 0, len(X_test)-1, 0)
    selected_student = X_test.iloc[student_idx]
    st.write("Student features:")
    st.dataframe(selected_student)

    # Convert Series to DataFrame
    student_df = selected_student.to_frame().T

    student_transformed = model.named_steps['preprocessor'].transform(student_df)
    shap_val_student = shap_values[1][student_idx][:len(all_features)]
    if len(shap_val_student) < len(all_features):
        shap_val_student = np.concatenate([shap_val_student, np.zeros(len(all_features) - len(shap_val_student))])

    shap_df_student = pd.DataFrame({
        "Feature": all_features,
        "SHAP Value": shap_val_student,
    }).sort_values("SHAP Value", ascending=True)

    fig_student = px.bar(
        shap_df_student,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title=f"SHAP Values for Student Index {student_idx}"
    )
    st.plotly_chart(fig_student)

    pred_class = model.predict(student_df)[0]
    st.write(f"**Predicted Class:** {'Pass' if pred_class==1 else 'Fail'}")

    # ---------------------------
    # 8. Multi-Student Risk Analysis
    # ---------------------------
    st.subheader("Top At-Risk Students")
    X_test['pred_prob'] = model.predict_proba(X_test)[:,0]
    X_test['pred_class'] = model.predict(X_test)
    X_test['G3_actual'] = y_test.values

    top_risk = X_test.sort_values('pred_prob', ascending=False).head(10)
    st.dataframe(top_risk[[*feature_cols,'G3_actual','pred_class','pred_prob']])

    st.markdown("**Interactive SHAP Analysis for Top At-Risk Students**")
    selected_risk_idx = st.selectbox("Select student index from top at-risk", top_risk.index)
    risk_student = X_test.loc[selected_risk_idx]

    risk_student_df = risk_student.to_frame().T
    shap_val_risk = shap_values[1][selected_risk_idx][:len(all_features)]
    if len(shap_val_risk) < len(all_features):
        shap_val_risk = np.concatenate([shap_val_risk, np.zeros(len(all_features) - len(shap_val_risk))])

    shap_df_risk = pd.DataFrame({
        "Feature": all_features,
        "SHAP Value": shap_val_risk,
    }).sort_values("SHAP Value", ascending=True)

    fig_risk = px.bar(
        shap_df_risk,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title=f"SHAP Values for At-Risk Student Index {selected_risk_idx}"
    )
    st.plotly_chart(fig_risk)
    pred_class_risk = model.predict(risk_student_df)[0]
    st.write(f"**Predicted Class:** {'Pass' if pred_class_risk==1 else 'Fail'}")
    st.write(f"**Fail Probability:** {risk_student['pred_prob']:.2f}")

else:
    st.info("Please upload a CSV file to start the analysis.")
