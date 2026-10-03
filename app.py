# ============================================================
# SCHOOL DROPOUT RISK PREDICTION
# Professional Streamlit Application
# REGRESSION VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
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


    # ========================================================
    # RISK SCORE CALCULATION
    # ========================================================

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

    failure_risk = (
        data["Previous_Failures"]
    )

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


    # ========================================================
    # CONTINUOUS TARGET
    # ========================================================

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

    data["Risk_Score"] = risk_score

    return data


# ============================================================
# TRAIN REGRESSION MODEL
# ============================================================

@st.cache_resource
def train_regression_model(data):

    features = [

        "Age",

        "Attendance_Percentage",

        "Average_Grade",

        "Study_Hours_Per_Week",

        "Family_Income",

        "Distance_From_School_KM",

        "Previous_Failures",

        "Disciplinary_Incidents",

        "Parental_Education_Years",

        "Internet_Access",

        "Extracurricular_Participation"
    ]


    X = data[features]

    y = data["Risk_Score"]


    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42
    )


    # ========================================================
    # STANDARDIZATION
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # RANDOM FOREST REGRESSOR
    # ========================================================

    model = RandomForestRegressor(

        n_estimators=250,

        max_depth=10,

        min_samples_split=5,

        min_samples_leaf=2,

        random_state=42
    )


    model.fit(

        X_train_scaled,

        y_train
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = model.predict(

        X_test_scaled
    )


    # ========================================================
    # REGRESSION METRICS
    # ========================================================

    mae = mean_absolute_error(

        y_test,

        y_pred
    )


    mse = mean_squared_error(

        y_test,

        y_pred
    )


    rmse = np.sqrt(mse)


    r2 = r2_score(

        y_test,

        y_pred
    )


    # MAPE can become unstable when actual values are near zero.
    # We calculate it safely using a small denominator.

    denominator = np.maximum(
        np.abs(y_test),
        1e-8
    )

    mape = np.mean(
        np.abs(
            (y_test - y_pred) /
            denominator
        )
    )


    metrics = {

        "mae": mae,

        "mse": mse,

        "rmse": rmse,

        "r2": r2,

        "mape": mape
    }


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame({

        "Feature":
            features,

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

    metrics,

    importance

) = train_regression_model(data)


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

            "📈 Regression Evaluation"
        ]
    )


    st.divider()


    st.caption(
        "Machine Learning Model"
    )

    st.caption(
        "Random Forest Regressor"
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


    # ========================================================
    # DASHBOARD METRICS
    # ========================================================

    # ========================================================
    # RISK SCORE DISTRIBUTION
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Risk Score Distribution"
        )


        fig, ax = plt.subplots(

            figsize=(7, 4)
        )


        sns.histplot(

            data=data,

            x="Risk_Score",

            kde=True,

            ax=ax
        )


        ax.set_xlabel(
            "Risk Score"
        )

        ax.set_ylabel(
            "Students"
        )


        st.pyplot(

            fig,

            use_container_width=True
        )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

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
        estimate a continuous student dropout-risk score.
x
        The regression model produces a numerical risk estimate
        rather than a binary dropout classification.
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
        "Enter student information below to generate a "
        "continuous model-based dropout risk score."
    )


    with st.form(
        "student_form"
    ):

        st.subheader(
            "Student Profile"
        )


        col1, col2, col3 = st.columns(3)


        # ====================================================
        # COLUMN 1
        # ====================================================

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


        # ====================================================
        # COLUMN 2
        # ====================================================

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


        # ====================================================
        # COLUMN 3
        # ====================================================

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

            "🔍 Predict Risk Score",

            use_container_width=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

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
                [
                    1
                    if internet == "Yes"
                    else 0
                ],

            "Extracurricular_Participation":
                [
                    1
                    if extracurricular == "Yes"
                    else 0
                ]
        })


        # ====================================================
        # SCALE INPUT
        # ====================================================

        scaled = scaler.transform(

            new_student
        )


        # ====================================================
        # REGRESSION PREDICTION
        # ====================================================

        predicted_risk = model.predict(

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

                "Predicted Risk Score",

                f"{predicted_risk:.2f}"
            )


        with col2:

            # -----------------------------------------------
            # Risk level is only a presentation aid
            # -----------------------------------------------

            if predicted_risk >= 3:

                st.markdown(

                    f"""
                    <div class="risk-high">

                    <h3>🔴 Higher Risk Score</h3>

                    <p>
                    The regression model predicts a risk score
                    of <strong>{predicted_risk:.2f}</strong>.
                    </p>

                    </div>
                    """,

                    unsafe_allow_html=True
                )


            elif predicted_risk >= 1:

                st.markdown(

                    f"""
                    <div class="risk-medium">

                    <h3>🟠 Moderate Risk Score</h3>

                    <p>
                    The regression model predicts a risk score
                    of <strong>{predicted_risk:.2f}</strong>.
                    </p>

                    </div>
                    """,

                    unsafe_allow_html=True
                )


            else:

                st.markdown(

                    f"""
                    <div class="risk-low">

                    <h3>🟢 Lower Risk Score</h3>

                    <p>
                    The regression model predicts a risk score
                    of <strong>{predicted_risk:.2f}</strong>.
                    </p>

                    </div>
                    """,

                    unsafe_allow_html=True
                )


        st.progress(

            min(
                max(
                    float(
                        (predicted_risk + 5) / 10
                    ),
                    0.0
                ),
                1.0
            )
        )


        st.caption(
            "The risk score is a model-generated continuous "
            "estimate and should not be interpreted as a definitive "
            "statement about a student's future."
        )


        # ====================================================
        # STUDENT INFORMATION
        # ====================================================

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

                "Age",

                "Attendance_Percentage",

                "Average_Grade",

                "Study_Hours_Per_Week",

                "Family_Income",

                "Distance_From_School_KM",

                "Previous_Failures",

                "Disciplinary_Incidents",

                "Parental_Education_Years",

                "Risk_Score"
            ]
        )


    with col2:

        chart_type = st.selectbox(

            "Chart type",

            [

                "Distribution",

                "Box Plot",

                "Risk Score Relationship"
            ]
        )


    # ========================================================
    # CHART
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(10, 5)
    )


    if chart_type == "Distribution":

        sns.histplot(

            data=data,

            x=selected_feature,

            kde=True,

            ax=ax
        )


        ax.set_title(

            f"Distribution of {selected_feature}"
        )


    elif chart_type == "Box Plot":

        sns.boxplot(

            data=data,

            y=selected_feature,

            ax=ax
        )


        ax.set_title(

            f"Box Plot of {selected_feature}"
        )


    else:

        if selected_feature == "Risk_Score":

            st.warning(
                "Select a feature other than Risk_Score "
                "for the relationship chart."
            )

        else:

            sns.scatterplot(

                data=data,

                x=selected_feature,

                y="Risk_Score",

                alpha=0.5,

                ax=ax
            )


            ax.set_title(

                f"{selected_feature} vs Risk Score"
            )

            ax.set_ylabel(
                "Risk Score"
            )


    st.pyplot(

        fig,

        use_container_width=True
    )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.subheader(

        "Dataset Preview"
    )


    st.dataframe(

        data.head(100),

        use_container_width=True
    )


    # ========================================================
    # DESCRIPTIVE STATISTICS
    # ========================================================

    st.subheader(

        "Descriptive Statistics"
    )


    st.dataframe(

        data.describe(),

        use_container_width=True
    )


# ============================================================
# REGRESSION EVALUATION
# ============================================================

elif page == "📈 Regression Evaluation":

    st.markdown(

        '<div class="section-heading">'
        '📈 Regression Evaluation'
        '</div>',

        unsafe_allow_html=True
    )


    st.write(

        "This section evaluates the Random Forest Regressor "
        "using standard regression evaluation metrics."
    )


    # ========================================================
    # METRICS
    # ========================================================

    st.subheader(

        "Regression Evaluation Metrics"
    )


    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(

            "MAE",

            f"{metrics['mae']:.3f}"
        )


    with col2:

        st.metric(

            "MSE",

            f"{metrics['mse']:.3f}"
        )


    with col3:

        st.metric(

            "RMSE",

            f"{metrics['rmse']:.3f}"
        )


    with col4:

        st.metric(

            "R² Score",

            f"{metrics['r2']:.3f}"
        )


    with col5:

        st.metric(

            "MAPE",

            f"{metrics['mape']:.2%}"
        )


    st.divider()


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.subheader(

        "Actual vs Predicted Risk Score"
    )


    fig, ax = plt.subplots(

        figsize=(9, 5)
    )


    ax.scatter(

        y_test,

        y_pred,

        alpha=0.6
    )


    minimum = min(

        y_test.min(),

        y_pred.min()
    )


    maximum = max(

        y_test.max(),

        y_pred.max()
    )


    ax.plot(

        [minimum, maximum],

        [minimum, maximum],

        linestyle="--"
    )


    ax.set_xlabel(

        "Actual Risk Score"
    )


    ax.set_ylabel(

        "Predicted Risk Score"
    )


    ax.set_title(

        "Actual vs Predicted Risk Score"
    )


    st.pyplot(

        fig,

        use_container_width=True
    )


    st.divider()


    # ========================================================
    # RESIDUAL ANALYSIS
    # ========================================================

    st.subheader(

        "Residual Analysis"
    )


    residuals = (

        y_test -

        y_pred
    )


    fig, ax = plt.subplots(

        figsize=(9, 5)
    )


    ax.scatter(

        y_pred,

        residuals,

        alpha=0.6
    )


    ax.axhline(

        0,

        linestyle="--"
    )


    ax.set_xlabel(

        "Predicted Risk Score"
    )


    ax.set_ylabel(

        "Residual"
    )


    ax.set_title(

        "Residual Plot"
    )


    st.pyplot(

        fig,

        use_container_width=True
    )


    st.divider()


    # ========================================================
    # ERROR DISTRIBUTION
    # ========================================================

    st.subheader(

        "Prediction Error Distribution"
    )


    fig, ax = plt.subplots(

        figsize=(9, 5)
    )


    sns.histplot(

        residuals,

        kde=True,

        ax=ax
    )


    ax.set_xlabel(

        "Prediction Error"
    )


    ax.set_ylabel(

        "Frequency"
    )


    ax.set_title(

        "Distribution of Residuals"
    )


    st.pyplot(

        fig,

        use_container_width=True
    )


    st.divider()


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(

        "Feature Importance"
    )


    st.dataframe(

        importance,

        use_container_width=True,

        hide_index=True
    )


    st.divider()


    # ========================================================
    # METRIC EXPLANATION
    # ========================================================

    st.subheader(

        "Metric Explanation"
    )


    st.markdown(
        """
        **MAE — Mean Absolute Error**

        Measures the average absolute difference between the
        actual and predicted risk scores. Lower values indicate
        smaller average prediction errors.


        **MSE — Mean Squared Error**

        Measures the average squared prediction error. Large
        errors have a greater influence on this metric.


        **RMSE — Root Mean Squared Error**

        The square root of MSE. It is expressed in the same
        units as the risk score.


        **R² Score**

        Measures the proportion of variation in the risk score
        explained by the regression model. Values closer to 1
        indicate that the model explains more of the variation.


        **MAPE — Mean Absolute Percentage Error**

        Measures prediction error as a percentage. Because
        percentage errors can become unstable when actual
        values are close to zero, this project calculates MAPE
        using a small denominator safeguard.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        EduGuard • School Dropout Risk Analytics<br>

        Data Thinking + Machine Learning • Regression Demonstration

    </div>
    """,

    unsafe_allow_html=True
)