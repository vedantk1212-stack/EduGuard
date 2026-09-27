# ============================================================
# SCHOOL DROPOUT RISK PREDICTION
# Professional Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduGuard | Dropout Risk Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* Cards */
    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }

    /* Risk cards */
    .risk-high {
        background: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 20px;
        border-radius: 12px;
    }

    .risk-medium {
        background: #fffbeb;
        border-left: 6px solid #d97706;
        padding: 20px;
        border-radius: 12px;
    }

    .risk-low {
        background: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 20px;
        border-radius: 12px;
    }

    /* Section heading */
    .section-heading {
        font-size: 25px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Info box */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA GENERATION
# ============================================================

@st.cache_data
def generate_data(n=2500):

    np.random.seed(42)

    data = pd.DataFrame({

        "Age": np.random.randint(14, 20, n),

        "Attendance_Percentage": np.clip(
            np.random.normal(78, 14, n),
            35,
            100
        ),

        "Average_Grade": np.clip(
            np.random.normal(65, 15, n),
            20,
            100
        ),

        "Study_Hours_Per_Week": np.clip(
            np.random.normal(10, 5, n),
            0,
            30
        ),

        "Family_Income": np.clip(
            np.random.normal(45000, 20000, n),
            5000,
            150000
        ),

        "Distance_From_School_KM": np.clip(
            np.random.exponential(4, n),
            0.2,
            30
        ),

        "Previous_Failures": np.clip(
            np.random.poisson(0.8, n),
            0,
            5
        ),

        "Disciplinary_Incidents": np.clip(
            np.random.poisson(0.6, n),
            0,
            5
        ),

        "Parental_Education_Years": np.clip(
            np.random.normal(11, 3, n),
            0,
            20
        ),

        "Internet_Access": np.random.binomial(
            1,
            0.75,
            n
        ),

        "Extracurricular_Participation": np.random.binomial(
            1,
            0.55,
            n
        )
    })

    # Risk calculations

    attendance_risk = (
        80 - data["Attendance_Percentage"]
    ) / 15

    grade_risk = (
        65 - data["Average_Grade"]
    ) / 15

    income_risk = (
        50000 - data["Family_Income"]
    ) / 20000

    distance_risk = (
        data["Distance_From_School_KM"]
    ) / 8

    failure_risk = data["Previous_Failures"]

    discipline_risk = (
        data["Disciplinary_Incidents"]
    ) / 2

    parent_risk = (
        12 - data["Parental_Education_Years"]
    ) / 4

    study_risk = (
        10 - data["Study_Hours_Per_Week"]
    ) / 5

    internet_risk = (
        1 - data["Internet_Access"]
    )

    activity_risk = (
        1 - data["Extracurricular_Participation"]
    )

    risk_score = (

        1.30 * attendance_risk +

        1.15 * grade_risk +

        0.60 * income_risk +

        0.45 * distance_risk +

        0.85 * failure_risk +

        0.55 * discipline_risk +

        0.35 * parent_risk +

        0.45 * study_risk +

        0.30 * internet_risk +

        0.20 * activity_risk +

        np.random.normal(0, 0.8, n)
    )

    probability = 1 / (
        1 + np.exp(-risk_score)
    )

    data["Dropout"] = np.random.binomial(
        1,
        probability
    )

    return data


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    X = data.drop(
        "Dropout",
        axis=1
    )

    y = data["Dropout"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = RandomForestClassifier(

        n_estimators=250,

        max_depth=10,

        min_samples_split=5,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_probability = model.predict_proba(
        X_test_scaled
    )[:, 1]

    metrics = {

        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "precision":
            precision_score(
                y_test,
                y_pred
            ),

        "recall":
            recall_score(
                y_test,
                y_pred
            ),

        "f1":
            f1_score(
                y_test,
                y_pred
            ),

        "auc":
            roc_auc_score(
                y_test,
                y_probability
            )
    }

    importance = pd.DataFrame({

        "Feature":
            X.columns,

        "Importance":
            model.feature_importances_
    }).sort_values(

        "Importance",

        ascending=False
    )

    return (
        model,
        scaler,
        X_test,
        y_test,
        y_pred,
        y_probability,
        metrics,
        importance
    )


# ============================================================
# LOAD DATA + MODEL
# ============================================================

data = generate_data()

(
    model,
    scaler,
    X_test,
    y_test,
    y_pred,
    y_probability,
    metrics,
    importance
) = train_model(data)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🎓 EduGuard"
    )

    st.markdown(
        "### Dropout Risk Analytics"
    )

    st.divider()

    page = st.radio(

        "Navigation",

        [
            "🏠 Dashboard",
            "🔮 Student Prediction",
            "📊 Data Analysis",
            "🤖 Model Performance",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.caption(
        "Machine Learning Model"
    )

    st.caption(
        "Random Forest Classifier"
    )

    st.caption(
        f"{len(data):,} simulated student records"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 EduGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Student Dropout Risk Prediction & Educational Analytics'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-heading">'
        '📈 Overview'
        '</div>',
        unsafe_allow_html=True
    )

    dropout_rate = (
        data["Dropout"].mean()
        * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    STUDENTS
                </div>
                <div class="metric-value">
                    {len(data):,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    DROPOUT RATE
                </div>
                <div class="metric-value">
                    {dropout_rate:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    MODEL ACCURACY
                </div>
                <div class="metric-value">
                    {metrics['accuracy']:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    ROC-AUC
                </div>
                <div class="metric-value">
                    {metrics['auc']:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Dropout Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.countplot(
            x="Dropout",
            data=data,
            ax=ax
        )

        ax.set_xlabel(
            "Dropout Status"
        )

        ax.set_ylabel(
            "Students"
        )

        ax.set_xticklabels(
            [
                "Remaining Enrolled",
                "Dropout"
            ]
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Top Predictive Factors"
        )

        top_features = importance.head(8)

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.barplot(

            data=top_features,

            x="Importance",

            y="Feature",

            ax=ax
        )

        ax.set_title(
            "Random Forest Feature Importance"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

    st.info(
        """
        **Data-thinking approach:** This dashboard uses academic,
        attendance, socioeconomic, and behavioral indicators to
        estimate dropout risk. In a real educational setting,
        predictions should support human intervention rather than
        automatically determine outcomes for students.
        """
    )


# ============================================================
# STUDENT PREDICTION
# ============================================================

elif page == "🔮 Student Prediction":

    st.markdown(
        '<div class="section-heading">'
        '🔮 Student Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter student information below to generate a model-based "
        "dropout risk estimate."
    )

    with st.form(
        "student_form"
    ):

        st.subheader(
            "Student Profile"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age",
                min_value=14,
                max_value=25,
                value=16
            )

            attendance = st.slider(
                "Attendance (%)",
                0,
                100,
                75
            )

            grade = st.slider(
                "Average Grade",
                0,
                100,
                65
            )

            study_hours = st.slider(
                "Study Hours / Week",
                0,
                40,
                10
            )

        with col2:

            income = st.number_input(
                "Family Income",
                min_value=5000,
                max_value=200000,
                value=45000,
                step=5000
            )

            distance = st.slider(
                "Distance from School (KM)",
                0.0,
                50.0,
                4.0,
                0.5
            )

            failures = st.number_input(
                "Previous Failures",
                min_value=0,
                max_value=10,
                value=0
            )

            discipline = st.number_input(
                "Disciplinary Incidents",
                min_value=0,
                max_value=10,
                value=0
            )

        with col3:

            parent_education = st.slider(
                "Parental Education (Years)",
                0,
                20,
                12
            )

            internet = st.selectbox(
                "Internet Access",
                ["Yes", "No"]
            )

            extracurricular = st.selectbox(
                "Extracurricular Participation",
                ["Yes", "No"]
            )

        submitted = st.form_submit_button(
            "🔍 Predict Dropout Risk",
            use_container_width=True
        )

    if submitted:

        new_student = pd.DataFrame({

            "Age": [age],

            "Attendance_Percentage":
                [attendance],

            "Average_Grade":
                [grade],

            "Study_Hours_Per_Week":
                [study_hours],

            "Family_Income":
                [income],

            "Distance_From_School_KM":
                [distance],

            "Previous_Failures":
                [failures],

            "Disciplinary_Incidents":
                [discipline],

            "Parental_Education_Years":
                [parent_education],

            "Internet_Access":
                [1 if internet == "Yes" else 0],

            "Extracurricular_Participation":
                [
                    1
                    if extracurricular == "Yes"
                    else 0
                ]
        })

        scaled = scaler.transform(
            new_student
        )

        probability = model.predict_proba(
            scaled
        )[0][1]

        probability_percentage = (
            probability * 100
        )

        prediction = model.predict(
            scaled
        )[0]

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        col1, col2 = st.columns(
            [1, 2]
        )

        with col1:

            st.metric(
                "Predicted Risk",
                f"{probability_percentage:.1f}%"
            )

        with col2:

            if probability >= 0.60:

                st.markdown(
                    f"""
                    <div class="risk-high">

                    <h3>🔴 Higher Predicted Risk</h3>

                    <p>
                    The model estimates a dropout probability
                    of <strong>
                    {probability_percentage:.1f}%
                    </strong>.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif probability >= 0.30:

                st.markdown(
                    f"""
                    <div class="risk-medium">

                    <h3>🟠 Moderate Predicted Risk</h3>

                    <p>
                    The model estimates a dropout probability
                    of <strong>
                    {probability_percentage:.1f}%
                    </strong>.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="risk-low">

                    <h3>🟢 Lower Predicted Risk</h3>

                    <p>
                    The model estimates a dropout probability
                    of <strong>
                    {probability_percentage:.1f}%
                    </strong>.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.progress(
            float(probability)
        )

        st.caption(
            "The probability is a model output, not a definitive "
            "statement about the student's future."
        )

        st.subheader(
            "Student Information"
        )

        display_student = new_student.copy()

        display_student.columns = [
            "Age",
            "Attendance (%)",
            "Average Grade",
            "Study Hours / Week",
            "Family Income",
            "Distance (KM)",
            "Previous Failures",
            "Disciplinary Incidents",
            "Parental Education",
            "Internet Access",
            "Extracurricular"
        ]

        st.dataframe(
            display_student,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.markdown(
        '<div class="section-heading">'
        '📊 Student Data Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_feature = st.selectbox(

            "Select a feature",

            [
                "Attendance_Percentage",
                "Average_Grade",
                "Study_Hours_Per_Week",
                "Family_Income",
                "Distance_From_School_KM",
                "Previous_Failures",
                "Disciplinary_Incidents",
                "Parental_Education_Years"
            ]
        )

    with col2:

        chart_type = st.selectbox(

            "Chart type",

            [
                "Distribution",
                "Box Plot"
            ]
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    if chart_type == "Distribution":

        sns.histplot(
            data=data,
            x=selected_feature,
            hue="Dropout",
            kde=True,
            ax=ax
        )

    else:

        sns.boxplot(
            data=data,
            x="Dropout",
            y=selected_feature,
            ax=ax
        )

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        data.head(100),
        use_container_width=True
    )

    st.subheader(
        "Descriptive Statistics"
    )

    st.dataframe(
        data.describe(),
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="section-heading">'
        '🤖 Machine Learning Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The application uses a Random Forest Classifier."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision']:.2%}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']:.2%}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics['f1']:.2%}"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{metrics['auc']:.2%}"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=[
                "Not Dropout",
                "Dropout"
            ],
            yticklabels=[
                "Not Dropout",
                "Dropout"
            ]
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "ROC Curve"
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            y_probability
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {metrics['auc']:.3f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "ROC Curve"
        )

        ax.legend()

        st.pyplot(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Feature Importance"
    )

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-heading">'
        'ℹ️ About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        <h3>🎯 Objective</h3>

        The objective of this project is to demonstrate how
        <strong>Data Thinking</strong> and
        <strong>Machine Learning</strong> can be combined to
        investigate factors associated with student dropout.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "Data Thinking Framework"
    )

    steps = {

        "1️⃣ Define":
            "Define student dropout as the social problem.",

        "2️⃣ Collect":
            "Collect academic, attendance, socioeconomic and behavioral data.",

        "3️⃣ Prepare":
            "Clean, transform and prepare the data.",

        "4️⃣ Analyze":
            "Explore relationships between student characteristics and dropout.",

        "5️⃣ Model":
            "Train a Random Forest classification model.",

        "6️⃣ Evaluate":
            "Evaluate accuracy, precision, recall, F1-score and ROC-AUC.",

        "7️⃣ Act":
            "Use predictions as signals for appropriate human support."
    }

    for title, description in steps.items():

        st.markdown(
            f"""
            **{title}**

            {description}
            """
        )

    st.divider()

    st.subheader(
        "Machine Learning Features"
    )

    feature_table = pd.DataFrame({

        "Feature": [
            "Attendance",
            "Average Grade",
            "Study Hours",
            "Family Income",
            "Distance to School",
            "Previous Failures",
            "Disciplinary Incidents",
            "Parental Education",
            "Internet Access",
            "Extracurricular Participation"
        ],

        "Type": [
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Binary",
            "Binary"
        ]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        """
        **Responsible-use note:** This application uses synthetic
        data for demonstration. In a real educational setting,
        student predictions require appropriate privacy protections,
        validation, fairness testing, and human oversight.
        A model should not automatically label, punish, exclude,
        or deny opportunities to a student.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        EduGuard • School Dropout Risk Analytics<br>
        Data Thinking + Machine Learning • Educational Demonstration
    </div>
    """,
    unsafe_allow_html=True
)