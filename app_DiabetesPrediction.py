# Best finalised model. EDA & SHAP done on 9/12/2025

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Capstone App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = r"final_rf_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

FEATURE_COLS = [
    'age',
    'bmi',
    'hbA1c_level',
    'blood_glucose_level',
    'hypertension',
    'heart_disease',
    'gender_Female',
    'gender_Male',
    'gender_Other'
]
# Load dataset once

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "diabetes_dataset.csv")
df = pd.read_csv(csv_path)

# Subset of diabetics for profiles (raw df names)
diabetes_df = df[df["diabetes"] == 1]
profile_cols = ["age", "hbA1c_level", "bmi", "blood_glucose_level"]
diab_profile = diabetes_df[profile_cols].mean()

# -----------------------------
# Prepare data for SHAP (model-side)
# -----------------------------
df_shap = df.copy()
df_shap["hba1c"] = df_shap["hbA1c_level"]
df_shap["glucose"] = df_shap["blood_glucose_level"]
df_shap["gender_F"] = (df_shap["gender"] == "Female").astype(int)
df_shap["gender_M"] = (df_shap["gender"] == "Male").astype(int)
df_shap["gender_O"] = (df_shap["gender"] == "Other").astype(int)

# Sidebar: patient form
with st.sidebar.form("patient_form"):
    st.header("Patient Information")

    age = st.number_input("Age (years)", min_value=1, max_value=120, value=40)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    gender_F = 1 if gender == "Female" else 0
    gender_M = 1 if gender == "Male" else 0
    gender_O = 1 if gender == "Other" else 0

    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0, max_value=60.0, value=25.0,
        help="BMI = weight (kg) / [height (m)]². Normal range is about 18.5–24.9."
    )
    hba1c = st.number_input(
        "HbA1c (%)",
        min_value=3.0, max_value=20.0, value=5.5,
        help="Average blood sugar level over ~3 months."
    )
    glucose = st.number_input(
        "Blood Glucose (mg/dL)",
        min_value=50, max_value=300, value=100,
        help="For fasting, normal is roughly 70–99 mg/dL."
    )

    hypertension = 1 if st.selectbox("Hypertension history", ["No", "Yes"]) == "Yes" else 0
    heart_disease = 1 if st.selectbox("Heart disease history", ["No", "Yes"]) == "Yes" else 0

    submitted = st.form_submit_button("Predict Diabetes Risk")
# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Diabetes Prediction", "Lifestyle Recommendations",
     "EDA & Visualizations", "Model Explainability"]
)
# ==========================
# TAB 1: DIABETES PREDICTION
# ==========================
with tab1:
    st.title("🩺 Diabetes Risk Prediction")
    st.caption("Educational tool to explore factors linked to diabetes risk. This is NOT a medical diagnosis.")

    prob_diabetes = None
    gauge_color = ""
    gauge_label = ""

    if not submitted:
        st.subheader("Step 1 – Enter patient information")
        st.info("Waiting for input…")
    else:
        # Prepare input for model
        input_data = np.array([[age, bmi, hba1c, glucose,
                                hypertension, heart_disease,
                                gender_F, gender_M, gender_O]])
        prediction = model.predict(input_data)[0]
        prob_diabetes = float(model.predict_proba(input_data)[0][1])

        # Assign risk label and color
        if prob_diabetes < 0.2:
            gauge_color = "green"
            gauge_label = "Low"
        elif prob_diabetes < 0.5:
            gauge_color = "orange"
            gauge_label = "Moderate"
        else:
            gauge_color = "red"
            gauge_label = "High"
        # NEW: share risk info with other tabs 10th dec
        st.session_state["risk_label"] = gauge_label
        st.session_state["risk_prob"] = prob_diabetes
    # ---------- EXPANDER 1: Prediction + Risk Gauge + Radar ----------
    if submitted and prob_diabetes is not None:
        with st.expander("📌 Diabetes Prediction 🚦Risk Gauge 👥 Profile Comparison", expanded=True):
            col_left, col_right = st.columns(2)

            # LEFT: Prediction + Risk Gauge (layout from code 1, colours from code 2)
            with col_left:
                st.markdown(
                    f"""
                    <div style="color:#4A90E2; font-size:20px; line-height:1.5;">
                        📌 <b>Prediction Result</b><br><br>
                        Estimated Probability of Diabetes:
                        <b style="color:{gauge_color};">{prob_diabetes*100:.1f}%</b><br>
                        Predicted Risk Level:
                        <b style="color:{gauge_color};">{gauge_label}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<h4 style='color:#4A90E2; font-size:20px;'>🚦Risk Gauge</h4>",
                    unsafe_allow_html=True
                )

                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_diabetes*100,
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    number={'font': {'size': 26, 'color': 'white'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white'},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 20],  'color': 'rgba(0, 128, 0, 0.45)'},   # darker green
                            {'range': [20, 50], 'color': 'rgba(255, 165, 0, 0.55)'}, # orange
                            {'range': [50, 100],'color': 'rgba(220, 20, 60, 0.60)'}, # crimson
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.8,
                            'value': prob_diabetes*100
                        }
                    }
                ))

                title = {
                'text': f"<span style='font-size:18px;'>Estimated Risk (%): </span>"
                    f"<span style='color:{gauge_color}; font-size:22px;'>{gauge_label}</span>",
    'font': {'color': 'white', 'size': 18}
}

                gauge_fig.update_layout(
                    title=title,
                    width=475, height=475,
                    margin=dict(t=50, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(gauge_fig, use_container_width=False)

                # Risk advisory
                if prob_diabetes < 0.2:
                    st.success("The model estimates a **low** diabetes risk based on your inputs.")
                    st.balloons()
                elif prob_diabetes < 0.5:
                    st.warning("The model estimates a **moderate** risk. Consider lifestyle changes and regular screening.")
                else:
                    st.error("The model estimates a **high** risk. Please discuss this with a healthcare professional.")

                st.caption("These values come from the machine learning model and are not a medical diagnosis.")

            # RIGHT: Radar chart (structure from code 1, labels/colours from code 2)
            with col_right:
                diab_profile_norm = (diab_profile - df[profile_cols].min()) / (
                    df[profile_cols].max() - df[profile_cols].min()
                )
                patient_profile = pd.Series({
                    "age": age,
                    "hbA1c_level": hba1c,
                    "bmi": bmi,
                    "blood_glucose_level": glucose
                })
                patient_norm = (patient_profile - df[profile_cols].min()) / (
                    df[profile_cols].max() - df[profile_cols].min()
                )
                # ADD THESE TWO LINES
                diab_r = diab_profile_norm.values
                patient_r = patient_norm.values
                # Pretty labels (same order as profile_cols)
                display_labels = ["Age", "HbA1c level", "BMI", "Blood glucose level"]

                st.markdown(
                    "<h4 style='color:#4A90E2; font-size:20px; margin-top:0px;'>👥 Profile Comparison (Radar)</h4>",
                    unsafe_allow_html=True
                )


                radar_fig = go.Figure()
                radar_fig.add_trace(go.Scatterpolar(
                    r=diab_r,
                    theta=profile_cols,           # use original keys for positions
                    fill="toself",
                    name="Typical Diabetic Patient",
                    line=dict(color="crimson")
                ))
                radar_fig.add_trace(go.Scatterpolar(
                    r=patient_r,
                    theta=profile_cols,
                    fill="toself",
                    name="Current Patient",
                    line=dict(color="dodgerblue")
                ))

                radar_fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            gridcolor="gray",
                            tickfont=dict(color="white")
                        ),
                        angularaxis=dict(
                            tickmode="array",
                            tickvals=profile_cols,      # underlying keys
                            ticktext=display_labels,    # what you see (no underscores, BMI)
                            tickfont=dict(color="white", size=12)
                        )
                    ),
                    showlegend=True,
                    title=dict(
                        text="Current Patient vs Typical Diabetes Profile (Dataset)",
                        font=dict(color="white", size=18)
                    ),
                    font=dict(size=14, color="white"),
                    width=480,
                    height=480,
                    margin=dict(t=130, b=40, l=40, r=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(radar_fig, use_container_width=True)



# ---------- EXPANDER 2: Patient Profile Summary ----------
    with st.expander("👤 Current Patient Profile Summary", expanded=False):
            patient_table = pd.DataFrame({
                "Feature": [
                    "Age", "Gender", "BMI", "HbA1c (%)", "Blood Glucose (mg/dL)",
                    "Hypertension history", "Heart disease history"
                ],
                "Value": [
                    age,
                    gender,
                    round(bmi, 1),
                    round(hba1c, 1),
                    round(glucose, 0),
                    "Yes" if hypertension else "No",
                    "Yes" if heart_disease else "No"
                ]
            })
            st.table(patient_table.set_index("Feature"))

 # ---------- EXPANDER 3: Feature Distributions ----------
    with st.expander("📈 Key Feature Distributions (Dataset vs Current Patient)", expanded=False):
            col_a, col_b, col_c, col_d = st.columns(4)

            def plot_feature_dist(ax, series, feature_name, patient_val, diabetic_mean):
                sns.histplot(series, bins=30, stat="density", ax=ax, color="#ADD8E6")
                sns.kdeplot(series, ax=ax, color="#0000CD", linewidth=1.5)
                ax.axvline(diabetic_mean, color="red", linestyle="-", label="Typical Diabetes Mean")
                ax.axvline(patient_val, color="blue", linestyle="--", label="Current Patient Metric")
                ax.set_title(f"{feature_name}", fontsize=11, fontweight="bold")
                ax.legend(loc="upper right", fontsize=6)
                ax.tick_params(labelsize=6)

            feature_keys   = ["age", "bmi", "hbA1c_level", "blood_glucose_level"]
            feature_labels = ["Age", "BMI", "HbA1c level", "Blood glucose level"]

            plots = zip([col_a, col_b, col_c, col_d],
                feature_keys,
                feature_labels,
                [age, bmi, hba1c, glucose])

            for col, feature_key, feature_label, patient_val in plots:
                fig, ax = plt.subplots(figsize=(3, 3))
                plot_feature_dist(ax, df[feature_key], feature_label, patient_val, diab_profile[feature_key])
                fig.tight_layout()
                col.pyplot(fig)


# ---------- EXPANDER 4: Clinical Metrics ----------
    with st.expander("🧬 Clinical Metrics (WHO / ADA) vs Current Patient Metrics", expanded=False):
            # BMI, HbA1c, Glucose categories
            bmi_cat = "Underweight" if bmi < 18.5 else "Normal weight" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            hba1c_cat = "Normal" if hba1c < 5.7 else "Prediabetes" if hba1c < 6.5 else "Diabetes"
            glu_cat = "Normal" if glucose < 100 else "Prediabetes" if glucose < 126 else "Diabetes"

            clinical_table = pd.DataFrame({
                "Metric": ["BMI", "HbA1c", "Fasting glucose"],
                "Your value": [f"{bmi:.1f}", f"{hba1c:.1f}%", f"{glucose:.0f} mg/dL"],
                "Clinical category": [bmi_cat, hba1c_cat, glu_cat]
            })
            st.table(clinical_table.set_index("Metric"))


                                   
            st.markdown("#### Clinical Metrics Distributions (Dataset vs Clinical cut-offs vs Patient)")

            col_cm1, col_cm2, col_cm3 = st.columns(3)

            def plot_clinical_dist(ax, series, feature_name, patient_val, cutoffs):
                sns.histplot(series, bins=30, stat="density", ax=ax, color="#ADD8E6")
                sns.kdeplot(series, ax=ax, color="#0000CD", linewidth=1.5)
                for c in cutoffs:
                    ax.axvline(c, color="red", linestyle="-", linewidth=1)
                ax.axvline(patient_val, color="blue", linestyle="--", linewidth=2, label="Patient")
                ax.set_title(feature_name)
                ax.legend()

            with col_cm1:
                fig_bmi_c, ax_bmi_c = plt.subplots()
                plot_clinical_dist(
                    ax_bmi_c,
                    df["bmi"],
                    "BMI (WHO bands)",
                    bmi,
                    cutoffs=[18.5, 25, 30]
                )
                st.pyplot(fig_bmi_c)

            with col_cm2:
                fig_hba1c_c, ax_hba1c_c = plt.subplots()
                plot_clinical_dist(
                    ax_hba1c_c,
                    df["hbA1c_level"],
                    "HbA1c (%) (ADA bands)",
                    hba1c,
                    cutoffs=[5.7, 6.5]
                )
                st.pyplot(fig_hba1c_c)

            with col_cm3:
                fig_glu_c, ax_glu_c = plt.subplots()
                plot_clinical_dist(
                    ax_glu_c,
                    df["blood_glucose_level"],
                    "Fasting glucose (mg/dL) (ADA bands)",
                    glucose,
                    cutoffs=[100, 126]
                )
                st.pyplot(fig_glu_c)

            # only run this logic when prob_diabetes is available
            if prob_diabetes is not None:
                if prob_diabetes < 0.2 and bmi < 25 and hba1c < 5.7 and glucose < 100:
                    st.success(
                        "Overall, your model-estimated diabetes risk is low and your BMI, HbA1c and glucose "
                        "are within normal ranges by WHO/ADA criteria. Maintain healthy habits."
                    )
                else:
                    st.warning(
                        "One or more of your values or the model-estimated risk is outside the normal range. "
                        "Please discuss these results with a healthcare professional."
                    )
    st.markdown("""
    <div style="border:1px solid #4FA3DB; border-radius:6px; padding:10px;">
    <b>Glossary:</b><br>
    • <b>BMI (Body Mass Index)</b>: weight (kg) divided by height (m²); commonly used to classify underweight, healthy weight, overweight and obesity.<br>
    • <b>HbA1c</b>: a blood test showing average blood sugar over the past ~3 months; higher values suggest poorer long‑term glucose control.<br>
    • <b>Blood glucose</b>: the level of sugar in the blood at a single point in time (e.g. fasting blood sugar in mg/dL).<br>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 2: LIFESTYLE
# ============================================================
with tab2:
    st.title("🥦 Lifestyle Recommendations")
    st.caption("General educational advice to support healthy blood sugar levels.")

    # Read risk info from tab 1 (if available)
    risk_label = st.session_state.get("risk_label", None)
    risk_prob = st.session_state.get("risk_prob", None)

    if risk_label is None or risk_prob is None:
        st.info(
            "Enter patient information and run a prediction in the **Diabetes Prediction** tab "
            "to see personalised lifestyle notes here."
        )
    else:
        if risk_label == "Low":
            st.success(
                f"Your model‑estimated diabetes risk is **low** (~{risk_prob*100:.1f}%). "
                "This is reassuring, and the main goal is to **keep it low over time** by protecting a "
                "healthy weight, staying active, and keeping blood sugar in a normal range."
            )
            st.write(
                "Use the recommendations below as general guidance to maintain healthy habits "
                "and continue routine health checks as advised by your clinician."
            )

        elif risk_label == "Moderate":
            st.warning(
                f"Your model‑estimated diabetes risk is **moderate** (~{risk_prob*100:.1f}%). "
                "Some of your current metrics (such as BMI, HbA1c or glucose) are drifting away from "
                "the ideal range in this dataset."
            )
            st.write(
                "The suggestions below focus on **tightening daily habits** around food, activity and "
                "weight management, and on **discussing regular screening** with a healthcare professional "
                "to prevent progression."
            )

        else:  # High
            st.error(
                f"Your model‑estimated diabetes risk is **high** (~{risk_prob*100:.1f}%). "
                "Several of your inputs (for example HbA1c, glucose, BMI or blood‑pressure‑related history) "
                "look concerning in this dataset."
            )
            st.write(
                "The recommendations below are **educational only** and should not delay medical care. "
                "Please consider using them as a starting point for discussion with a doctor or nurse, who "
                "can order proper tests and give personalised advice."
            )

        # --- Risk‑specific lifestyle content (replaces old shared bullets) ---
        if risk_label == "Low":
            st.subheader("Nutrition Therapy")
            st.write(
                "- At your current low predicted risk, the aim is to **keep this risk low over time**.\n"
                "- Focus on a generally balanced pattern: more low glycemic index foods (such as whole grains, "
                "beans, fruit and non-starchy vegetables) and anti-inflammatory choices like colourful vegetables, "
                "nuts, seeds and healthy oils in everyday meals.\n"
                "- You may not need a strict meal plan but thinking about a loose, personalised pattern that fits your "
                "usual cuisine can help to maintain weight, cholesterol and blood sugar in a healthy range."
            )

            st.subheader("Lifestyle Coaching")
            st.write(
                "- Lifestyle habits at this stage are about **supporting long-term health**, not treating a disease.\n"
                "- Aim for regular movement that feels realistic (for example, walks, light cycling or simple strength "
                "exercises a few times a week).\n"
                "- Build in small stress-management habits (relaxation, hobbies, time with people you enjoy) and try to "
                "keep fairly regular sleep and wake times.\n"
                "- The goal is to find routines you can genuinely live with, so feeling well and staying active become "
                "part of your normal lifestyle."
            )

            st.subheader("Monitoring")
            st.write(
                "- With low risk, Monitoring usually means **routine health checks** rather than frequent tests.\n"
                "- This can include checking weight, blood pressure and cholesterol and having blood tests like fasting "
                "glucose or HbA1c at intervals recommended by your clinician, especially if you have family history or "
                "other risk factors.\n"
                "- Most people at low risk do not need to check their own blood sugar or use Continuous Glucose "
                "Monitoring Systems (CGMS – small sensors worn on the skin that track glucose every few minutes). "
                "These tools are usually reserved for higher risk or confirmed diabetes and can be discussed if your "
                "situation changes."
            )

            st.subheader("Education and remission mindset")
            st.write(
                "- Use this low-risk result as a reminder to **stay ahead of problems**.\n"
                "- Learning how to read food labels (serving size, total carbohydrates, added sugars, fibre) and knowing "
                "your personal risk factors (for example, family history, central weight gain, high blood pressure or "
                "lipids) helps you recognise when to seek advice.\n"
                "- Think of remission here as never letting diabetes develop: working with a clinician to review "
                "medications if needed and to confirm that your nutrition and activity plan are adequate for the long term."
            )

        elif risk_label == "Moderate":
            st.subheader("Nutrition Therapy")
            st.write(
                "- A moderate predicted risk suggests there is a **good opportunity to prevent or delay diabetes** with "
                "more structured nutrition changes.\n"
                "- This often includes shifting towards low glycemic index, higher-fibre meals (more vegetables, whole "
                "grains, legumes and fruits; fewer refined carbohydrates and sugary drinks) and using anti-inflammatory "
                "patterns such as Mediterranean-style or similar, adapted to your culture.\n"
                "- A personalised meal plan with a dietitian can target modest weight loss if needed, realistic portion "
                "sizes and practical swaps within your usual meals so changes are sustainable."
            )

            st.subheader("Lifestyle Coaching")
            st.write(
                "- Lifestyle Coaching at this level is more **structured and goal-focused**.\n"
                "- Many people aim for at least 150 minutes per week of moderate exercise plus 2–3 sessions of resistance "
                "training, adjusted to their fitness and health conditions.\n"
                "- Coaching can also cover stress management (for example, guided relaxation or counselling when needed) "
                "and sleep routines (a fairly consistent 7–9 hours), with practical steps that fit into daily life rather "
                "than feeling like an extra burden."
            )

            st.subheader("Monitoring")
            st.write(
                "- Monitoring becomes a bit more deliberate when risk is moderate.\n"
                "- Your clinician may suggest tracking weight, waist circumference and blood pressure and repeating "
                "glucose or HbA1c tests every 6–12 months, depending on your overall profile.\n"
                "- Some people at this risk level use occasional home blood sugar checks; in selected cases, a CGMS "
                "(Continuous Glucose Monitoring System – a small sensor worn on the skin that continuously tracks glucose "
                "and displays patterns) may be used under supervision to show how meals and activity affect glucose."
            )

            st.subheader("Education and physician-supervised remission")
            st.write(
                "- Education now emphasises that **type 2 diabetes is not inevitable** at this stage.\n"
                "- Learning to read food labels, understand personal risk factors and recognise the idea of "
                "'prediabetes' helps you see why acting now matters.\n"
                "- Physician-supervised remission here means working proactively with a clinician on lifestyle-first "
                "strategies, regular follow-up and protocol-based review of medications for blood pressure and lipids, "
                "so that improvements in weight, diet and activity can translate into lower long-term risk."
            )

        else:  # High
            st.subheader("Nutrition Therapy")
            st.write(
                "- A high predicted risk means your profile looks similar to people who often have or soon develop "
                "diabetes, so **intensive Nutrition Therapy with professional support is recommended**.\n"
                "- Personalised meal plans usually emphasise low glycemic index and anti-inflammatory foods, with careful "
                "attention to total energy, carbohydrate quality and heart health.\n"
                "- For some people, more structured programmes (including energy-restricted or meal-replacement-based "
                "approaches) may be considered to support weight loss and possible remission, but these must be done under "
                "medical supervision, especially if you take medicines that can cause low blood sugar."
            )

            st.subheader("Lifestyle Coaching")
            st.write(
                "- Lifestyle Coaching at high risk is more intensive and closely linked to medical care.\n"
                "- This can include a written exercise prescription tailored to your fitness level and other conditions, "
                "with gradual progression in aerobic and resistance training and clear safety advice.\n"
                "- Coaching should also address emotional and practical barriers, including stress, mood, and sleep, with "
                "referrals to psychological or social support if needed."
            )

            st.subheader("Monitoring")
            st.write(
                "- Monitoring at high risk is usually **more frequent and detailed**.\n"
                "- Clinicians often schedule regular blood tests for glucose, HbA1c, kidney function and lipids, along "
                "with blood pressure and weight checks, following guideline-based schedules.\n"
                "- Daily blood sugar self-monitoring is often recommended and CGMS devices (Continuous Glucose Monitoring "
                "Systems – small wearable sensors that measure glucose every few minutes and show trend arrows) can help "
                "you and your care team fine-tune medications, food and activity."
            )

            st.subheader("Education and physician-supervised remission protocols")
            st.write(
                "- Education here is similar to **ongoing diabetes self-management training**, even if a formal diagnosis "
                "has not yet been confirmed.\n"
                "- This includes learning to interpret food labels, match carbohydrate quality and quantity to activity "
                "and treatment, understand risks to the heart, kidneys, eyes and nerves and recognise symptoms that "
                "need urgent care.\n"
                "- **Physician-supervised remission** means having a clear plan with your clinician: protocol-driven reviews "
                "of medications (including when it may be safe to adjust doses), lipid and blood-pressure management and "
                "a written **Nutrition Therapy and exercise programme** with regular reviews to track progress over time."
            )

    with st.expander("🌏 Resources and support in Singapore", expanded=False):
        # keep your existing Singapore resources text here (unchanged)
        st.write(
            "- **National initiatives**: Singapore has declared a ‘War on Diabetes’ and runs many public programmes "
            "to support early screening, lifestyle change and good control of diabetes."
        )
        st.write(
            "- **Health Promotion Board (HPB)**: Look out for HPB campaigns, community exercise classes and "
            "healthy eating programmes in your neighbourhood (e.g. free or low‑cost workouts, mall roadshows)."
        )
        st.write(
            "- **ActiveSG & community sports facilities**: Community gyms and pools offer affordable ways to "
            "meet the 150 minutes/week physical activity goal."
        )
        st.write(
            "- **Primary care (GP / polyclinics)**: If your values or risk look high, book an appointment with "
            "a GP or polyclinic for proper blood tests and advice."
        )
        st.write(
            "- **Dietitian / nutritionist**: A dietitian can help tailor a meal plan to your culture, budget and "
            "health goals (for example, healthier versions of local favourites)."
        )
        st.write(
            "- **Education**: Local hospitals and organisations such as Diabetes Singapore often run talks, "
            "support groups and workshops on living well with or preventing diabetes."
        )
        st.caption(
            "Online videos (e.g. on YouTube) can be useful for ideas—try searching for "
            "'beginner low‑impact cardio', 'diabetes‑friendly meals', or 'healthy hawker choices in Singapore'—"
            "but always cross‑check with a healthcare professional."
        )

with tab3:
    st.title("📊 Exploratory Data Analysis & Visualizations")
    st.caption("Explore patterns, trends, and relationships within the training dataset using clean, interactive, Plotly-based charts.")

    # ===============================
    # DATA PREVIEW
    # ===============================
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    st.markdown(
        """
        <div style="text-align:center;">
        <b>Interpretation:</b><br><br>
        <b>Strongest Correlations</b><br>
        • <b>Diabetes & Blood Glucose Level (~0.4):</b> Higher glucose strongly increases diabetes likelihood.<br>
        • <b>Diabetes & HbA1c Level (~0.4):</b> Elevated HbA1c values align with higher diabetes prevalence.<br>
        • <b>Age & BMI (~0.3):</b> Older individuals tend to have higher BMI.<br><br>

        <b>Moderate Correlations</b><br>
        • <b>Hypertension & Age (~0.2):</b> A moderate correlation indicates that hypertension is more frequently observed among older individuals.<br>
        • <b>Heart Disease & Age (~0.2):</b> A moderate correlation suggests that heart disease tends to be more prevalent with increasing age.<br>

        <b> These correlations indicate associations, not causality.<b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ===============================
    # DIABETES DISTRIBUTION (PLOTLY)
    # ===============================
    st.subheader("📉 Diabetes Distribution")

    fig_dist = px.histogram(
        df,
        x="diabetes",
        color="diabetes",
        color_discrete_sequence=["#4FA3DB", "#E57357"],
        text_auto=True,
        height=350
    )
    fig_dist.update_layout(showlegend=False, bargap=0.3)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown(
    """
    <div style="text-align:center;">
    <b>Interpretation:</b><br>
    Diabetes cases account for only about 8–10% of the dataset, so the positive (diabetes) class is much smaller than the negative class.<br>
    This class imbalance can lead models to miss diabetics unless techniques such as class weighting or SMOTE are used to rebalance training data.
    </div><br>
    """,
    unsafe_allow_html=True
)


    st.markdown("---")

    # ===============================
    # CORRELATION HEATMAP (PX.IMSHOW)
    # ===============================
    st.subheader("📊 Correlation Heatmap (Interactive)")

    numeric_df = df.select_dtypes(include=np.number)
    fig_corr = px.imshow(
        numeric_df.corr(),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        height=500
    )
    fig_corr.update_layout(title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown(
    """
    <div style="text-align:center;">
    <b>Interpretation:</b><br>
    Glucose and HbA1c show the strongest positive links with diabetes in this dataset, meaning higher values tend to appear more often in people with diabetes.<br>
    Age and BMI also have moderate positive relationships with diabetes and with each other, reflecting that older individuals in this sample are more likely to have higher BMI and diabetes.<br>
    These are associations within this dataset and do not prove causality, but they align with clinical understanding of diabetes risk factors.
    </div>
    """,
    unsafe_allow_html=True
)


    st.markdown("---")

    # ===============================
    # FEATURE DISTRIBUTIONS – 1 ROW (PLOTLY)
    # ===============================
    st.subheader("📈 Feature Distributions")

    features = ["age", "bmi", "hbA1c_level", "blood_glucose_level"]
    cols = st.columns(4)

    for i, feat in enumerate(features):
        with cols[i]:
            fig = px.histogram(
                df,
                x=feat,
                nbins=30,
                color_discrete_sequence=["#4FA3DB"],
                marginal="box",
                height=300
            )
            fig.update_layout(title=feat.capitalize())
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
    """
    <div style="text-align:center;">
    <b>Interpretation:</b><br>
    HbA1c and blood glucose have right‑skewed distributions, with a subset of people showing much higher values than the rest of the population.<br>
    Age and BMI are more spread out, reflecting natural variability and together these four variables drive the model's predictive power to distinguish diabetic from non‑diabetic cases.
    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown("---")

    # ===============================
    # CLINICAL VARIABLES – 3 WIDE ROW (PLOTLY)
    # ===============================
    st.subheader("📦 Clinical Variables by Diabetes Status")

    clinical_vars = ["bmi", "hbA1c_level", "blood_glucose_level"]
    cols2 = st.columns(3)

    for i, feat in enumerate(clinical_vars):
        with cols2[i]:
            fig = px.box(
                df,
                x="diabetes",
                y=feat,
                color="diabetes",
                color_discrete_sequence=["#4FA3DB", "#E57357"],
                height=300
            )
            fig.update_layout(showlegend=False, title=feat.capitalize())
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
    """
    <div style="text-align:center;">
    <b>Interpretation:</b><br>
    Across BMI, HbA1c and blood glucose, the median and upper ranges are clearly higher for people with diabetes than for those without diabetes.<br>
    HbA1c and glucose show the clearest separation between groups, while BMI is moderately higher among diabetics, supporting their strong roles as predictors in the model.
    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown("---")

    # ===============================
    # PAIRPLOT → REPLACED WITH PLOTLY SCATTER MATRIX
    # ===============================
    st.subheader("🔗 Pairwise Relationships (Interactive)")

    sample_df = df.sample(400, random_state=42)
    sample_df["diabetes_str"] = sample_df["diabetes"].astype(str)

    fig_matrix = px.scatter_matrix(
        sample_df,
        dimensions=["age", "bmi", "hbA1c_level", "blood_glucose_level"],
        color="diabetes_str",
        color_discrete_map={
        "0": "#4FA3DB",      # Blue for non-diabetic
        "1": "#E57357"       # Orange for diabetic
    },
        height=700
    )
    fig_matrix.update_traces(diagonal_visible=False)
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🧬 Diabetes Profile Overview")

    # Split diabetics vs non‑diabetics
    df_diab = df[df["diabetes"] == 1]
    df_nondi = df[df["diabetes"] == 0]

# Core clinical features
    profile_cols = ["age", "bmi", "hbA1c_level", "blood_glucose_level"]

# Means for each group
    diab_mean = df_diab[profile_cols].mean()
    nondi_mean = df_nondi[profile_cols].mean()

# Tidy DataFrame for Plotly
    profile_df = pd.DataFrame({
    "Feature": ["Age", "BMI", "HbA1c level", "Blood glucose level"],
    "Diabetic mean": diab_mean.values,
    "Non‑diabetic mean": nondi_mean.values,
    })

    profile_long = profile_df.melt(
    id_vars="Feature",
    value_vars=["Diabetic mean", "Non‑diabetic mean"],
    var_name="Group",
    value_name="Value"
    )

    fig_profile = px.bar(
    profile_long,
    x="Feature",
    y="Value",
    color="Group",
    barmode="group",
    text="Value",
    color_discrete_sequence=["#E57357", "#4FA3DB"],  # diabetics = orange/red, non‑diabetics = blue
    height=400
    )
    fig_profile.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_profile.update_layout(
    title="Average Clinical Profile: Diabetic vs Non‑diabetic",
    yaxis_title="Mean value",
    xaxis_title="Feature",
    legend_title="Group",
    margin=dict(t=80, b=40, l=40, r=40)
    )
    # compute max value from your data
    max_val = profile_long["Value"].max()

    fig_profile.update_layout(
        title="Average Clinical Profile: Diabetic vs Non-diabetic",
        yaxis_title="Mean value",
        xaxis_title="Feature",
        legend_title="Group",
        margin=dict(t=80, b=40, l=40, r=40)
    )

# add headroom above tallest bar
    fig_profile.update_yaxes(range=[0, max_val * 1.25])

    st.plotly_chart(fig_profile, use_container_width=True)

# Inline interpretation text
    st.markdown("""
    <div style="text-align:center;">
    <b>Interpretation:</b><br>
    On average, people with diabetes in this dataset have higher HbA1c, blood glucose and BMI, and tend to be older, compared with non‑diabetic individuals.<br>
    These differences highlight why the model relies strongly on HbA1c and glucose, with BMI and age contributing additional signal.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📊 Diabetes Prevalence by Age Band")

# Age bands (quartiles; mirror your notebook logic if different)
    df["age_band"] = pd.qcut(
    df["age"],
    q=4,
    labels=["Youngest 25%", "25–50%", "50–75%", "Oldest 25%"]
    )

    age_counts = (
    df.groupby(["age_band", "diabetes"])
      .size()
      .reset_index(name="count")
    )

# Convert counts to percentages within each age band
    age_totals = age_counts.groupby("age_band")["count"].transform("sum")
    age_counts["percent"] = 100 * age_counts["count"] / age_totals

    age_counts["diabetes_str"] = age_counts["diabetes"].map({0: "No diabetes", 1: "Diabetes"})

    fig_age = px.bar(
        age_counts,
        x="age_band",
        y="percent",
        color="diabetes_str",
        barmode="stack",
        text="percent",
        color_discrete_sequence=["#4FA3DB", "#E57357"],
        height=400
    )
    fig_age.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig_age.update_layout(
        title="Diabetes prevalence within each age band",
        xaxis_title="Age band (quartiles)",
        yaxis_title="Percentage of people in band",
        legend_title="Status",
        yaxis=dict(range=[0, 100]),
        margin=dict(t=60, b=60, l=40, r=40)
    )

    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("""
    <div style="text-align:center;">
    <b>Interpretation:</b><br>
    Diabetes is present in every age band, but the proportion of diabetics grows in the older age groups, especially in the oldest 25% of the population.<br>
    Even in older bands, non‑diabetic individuals still form the majority, which highlights that age is an important risk factor but not a determinant on its own.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # ===============================
    # KEY INSIGHTS
    # ===============================
    
    st.subheader("📌 Key Insights from EDA")
    st.markdown(
    """
    <div style="
        border: 2px solid #4FA3DB;
        border-radius: 8px;
        padding: 14px;
        background-color: #1E1E1E;
        color: #E0E0E0;
        font-size: 14px;
        line-height: 1.6;">
    <b>Key insights from this dataset:</b><br>
    • The dataset has no missing values and shows a clear class imbalance, with diabetes cases making up less than 10% of all records.<br>
    • HbA1c and blood glucose are the strongest signals for diabetes in this dataset; higher values are frequently seen in people with diabetes.<br>
    • Age and BMI also trend higher in people with diabetes and act as important supporting features for the model.<br>
    • Because diabetes cases are much fewer than non-diabetes cases, rebalancing methods such as SMOTE will be useful to help the model learn the minority (diabetes) class better.<br>
    </div>
    """,
    unsafe_allow_html=True
    )

    
# ===============================
# TAB 4: Model Explainability & Performance
# ===============================


import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("final_rf_model.pkl")  # this is best_rf

# ===============================
# TAB 4: Model Explainability & Performance
# ===============================
with tab4:
    st.title("🧠 Model Explainability")
    st.caption("Understand why the Random Forest model was selected and how it makes predictions.")

    # ----------------------------------------------------
    # 1) MODEL FAMILY COMPARISON (WHY RANDOM FOREST?)
    # ----------------------------------------------------
    st.subheader("Model family comparison (baseline models)")

    # Replace these with your actual notebook results if different
    model_compare_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "XGBoost",
            "Explainable Boosting Machine",
            "Random Forest"
        ],
        "Accuracy": [0.95, 0.96, 0.96, 0.9776],
        "Recall (class 1)": [0.70, 0.74, 0.75, 0.97],
        "F1-score (class 1)": [0.78, 0.80, 0.81, 0.98]
    })

    st.dataframe(model_compare_df.style.format(precision=3), use_container_width=True)

    model_long = model_compare_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Recall (class 1)", "F1-score (class 1)"],
        var_name="Metric",
        value_name="Score"
    )

    fig_model_compare = px.bar(
        model_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        height=450,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_model_compare.update_traces(texttemplate="%{text:.3f}", textposition="inside", textfont_size=10)
    fig_model_compare.update_layout(
        title="Baseline models: performance on diabetes classification",
        yaxis=dict(range=[0, 1]),
        xaxis_tickangle=-20,
        margin=dict(b=100)
    )
    st.plotly_chart(fig_model_compare, use_container_width=True)

    st.markdown(
        """
        <div style="border:1px solid #4FA3DB; border-radius:6px; padding:10px; text-align:center;">
        <b>Interpretation:</b><br>
        All models perform reasonably well, but Random Forest achieves the strongest balance of recall and F1-score
        for the diabetes (positive) class while keeping very high overall accuracy. This is why Random Forest was
        selected as the final model for deployment.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ----------------------------------------------------
    # 2) RANDOM FOREST TRAINING EVOLUTION (PIPELINE)
    # ----------------------------------------------------
    st.subheader("Random Forest training pipeline: Impact of SMOTE, tuning and CV")

    perf_df = pd.DataFrame({
        "Model version": [
            "Before SMOTE",
            "After SMOTE",
            "After SMOTE + Tuning",
            "After SMOTE + Tuning + CV"
        ],
        "Accuracy": [0.96835, 0.9618, 0.96185, 0.97758],
        "Precision (class 1)": [0.91, 0.81, 0.81, 0.98],
        "Recall (class 1)": [0.69, 0.72, 0.72, 0.97],
        "F1-score (class 1)": [0.79, 0.76, 0.76, 0.98]
    })

    st.dataframe(perf_df.style.format(precision=3), use_container_width=True)

    perf_long = perf_df.melt(
        id_vars="Model version",
        value_vars=["Accuracy", "Precision (class 1)", "Recall (class 1)", "F1-score (class 1)"],
        var_name="Metric",
        value_name="Score"
    )

    fig_perf = px.bar(
        perf_long,
        x="Model version",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        height=500,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_perf.update_traces(texttemplate="%{text:.2f}", textposition="inside", textfont_size=10)
    fig_perf.update_layout(
        title="Random Forest metrics across pipeline versions",
        yaxis=dict(range=[0, 1]),
        xaxis_tickangle=-30,
        margin=dict(b=120)
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown(
    """
    <div style="
        border: 2px solid #4FA3DB;
        border-radius: 8px;
        padding: 14px;
        background-color: #1E1E1E;
        color: #E0E0E0;
        font-size: 14px;
        line-height: 1.6;">
        <b>Interpretation (Random Forest training pipeline):</b><br>
         <b>Baseline Random Forest</b>: Trained on the original imbalanced data, where non-diabetic cases are much more common than diabetic cases, so the model may miss some people with diabetes.<br>
         <b>+ SMOTE</b>: SMOTE (Synthetic Minority Over-sampling Technique) creates additional synthetic diabetes cases so the model sees more positive examples and becomes better at recognising the minority class.<br>
         <b>+ Hyperparameter tuning</b>: Key Random Forest settings (such as tree depth and number of trees) are adjusted to improve accuracy and recall, especially for the diabetes class.<br>
         <b>+ Cross-validation (CV)</b>: The tuned model is evaluated across several different train–test splits to check that the performance is stable and not just due to a single random split, making the final model more reliable for new patients.<br>
         <b>+ Before SMOTE, the model has high precision but lower recall, meaning many diabetics are missed.<br>
             <b>+ After applying SMOTE and tuning, recall for class 1 improves while precision stays acceptable.<br>
             <b>+ The final Random Forest (SMOTE + tuning + cross‑validation) delivers the best and most stable performance with Accuracy ≈ 0.978 and F1-score ≈ 0.98 for the diabetes class.<br>
        <b>+ Overall, moving from the baseline model to the final SMOTE + tuned + CV model improves how well the Random Forest detects diabetes while keeping overall accuracy strong.
        </div>
        """,
    unsafe_allow_html=True
    )


    st.markdown("---")

    # ----------------------------------------------------
    # 3) RANDOM FOREST FEATURE IMPORTANCE (GLOBAL)
    # ----------------------------------------------------
    st.subheader("Random Forest feature importance")

    features = FEATURE_COLS
    importances = model.feature_importances_
    # Pretty labels (remove underscores, capitalise BMI, etc.)
    pretty_name_map = {
        "age": "Age",
        "bmi": "BMI",
        "hbA1c_level": "HbA1c level",
        "blood_glucose_level": "Blood glucose level",
        "hypertension": "Hypertension",
        "heart_disease": "Heart disease",
        "gender_Female": "Gender Female",
        "gender_Male": "Gender Male",
        "gender_Other": "Gender Other",
    }

    feat_imp = pd.DataFrame({
    "Feature_raw": features,
    "Feature": [pretty_name_map.get(f, f) for f in features],
    "Importance": importances
    }).sort_values("Importance", ascending=False)

    fig_feat = px.bar(
        feat_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color="Importance",
        color_continuous_scale="Viridis",
        height=400
    )
    fig_feat.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_feat.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        title="Random Forest feature importance (Gini-based)"
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown(
        """
        <div style="border:1px solid #4FA3DB; border-radius:6px; padding:10px; text-align:center;">
        <b>Interpretation:</b><br>
        Features with larger bars contribute more to the Random Forest’s split decisions overall.<br>
        Glucose and HbA1c dominate the predictive signal, followed by BMI and age, which aligns with
        established clinical understanding of diabetes risk factors.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ----------------------------------------------------
    # 4) SHAP GLOBAL EXPLANATIONS (ALREADY MODEL-READY DF)
    # ----------------------------------------------------
    st.subheader("SHAP Feature Impact (Random Forest)")
    st.caption(
    "SHAP (SHapley Additive exPlanations) shows how much each feature pushes an individual prediction "
    "towards or away from diabetes. SHAP values here are computed on a sample of the dataset for speed."
)
    # --- Step 1: Create gender dummy columns ---
    if not all(col in df.columns for col in ["gender_Female", "gender_Male", "gender_Other"]):
        df["gender_Female"] = (df["gender"] == "Female").astype(int)
        df["gender_Male"]   = (df["gender"] == "Male").astype(int)
        df["gender_Other"]  = (df["gender"] == "Other").astype(int)

    st.subheader("SHAP Feature Impact")
    st.caption("Features pushing predictions higher (towards diabetes) or lower.")

    with st.expander("View SHAP summary plots (sample of data)", expanded=False):

        FEATURE_COLS_RF = [
            "age", "bmi", "hbA1c_level", "blood_glucose_level",
            "hypertension", "heart_disease",
            "gender_Female", "gender_Male", "gender_Other",
        ]

        FEATURE_LABELS_RF = [
            "Age",
            "BMI",
            "HbA1c level",
            "Blood glucose level",
            "Hypertension",
            "Heart disease",
            "Gender Female",
            "Gender Male",
            "Gender Other",
        ]
        if len(df) == 0:
            st.warning("No data available to compute SHAP values.")
        else:
            n_samples = min(200, len(df))
            X_shap = df[FEATURE_COLS_RF].sample(n=n_samples, random_state=42)

        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_shap, check_additivity=False)

        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            shap_values_class1 = shap_values[:, :, 1]

                 

            # Beeswarm plot
        st.write("Beeswarm plot (feature impact across individuals):")
        fig_bee, _ = plt.subplots(figsize=(8, 4))
        shap.summary_plot(
                shap_values_class1,
                X_shap,
                plot_type="dot",
                feature_names=FEATURE_LABELS_RF,
                show=False,
            )
        st.pyplot(fig_bee)
        st.markdown(
            """
            <div style="border:1px solid #4FA3DB; border-radius:6px; padding:10px;">
             <b>Interpretation (Beeswarm plot – feature impact):</b><br>
            Each dot is one patient; dots to the right increase the model’s predicted diabetes risk and dots to the left decrease it.<br>
            High values of HbA1c and glucose (red dots) mainly appear on the right, showing they strongly push predictions towards diabetes, while age, BMI and gender also move the prediction but with smaller or more mixed effects depending on the overall profile.<br>
            SHAP is important because it explains <i>why</i> the model makes a prediction for many individuals at once, and it shows that the model’s behaviour (for example, high HbA1c and glucose increasing risk) is consistent with clinical expectations, which helps build trust in the model.
            </div>
            """,
            unsafe_allow_html=True,
            )

            # Bar plot of mean absolute SHAP values
        st.write("Mean absolute SHAP values (global importance):")
        fig_bar, _ = plt.subplots(figsize=(8, 4))
        shap.summary_plot(
                shap_values_class1,
                X_shap,
                plot_type="bar",
                feature_names=FEATURE_LABELS_RF,
                show=False,
            )
        st.pyplot(fig_bar)

        st.markdown(
            """
            <div style="border:1px solid #4FA3DB; border-radius:6px; padding:10px;">
            <b>Interpretation (SHAP bar plot – mean absolute values):</b><br>
            Features with higher mean absolute SHAP values contribute more, on average, to shifting predictions up or down across all patients.<br>
            HbA1c has the largest mean SHAP value, followed by blood glucose, gender (Male), age, gender (Female), and then BMI, so the model’s diabetes risk estimate is driven most by HbA1c and glucose, with gender and age also influencing the prediction and BMI adding a smaller but still meaningful effect.<br>
            These values describe how this Random Forest behaves on this dataset and do not replace clinical judgement about which risk factors are medically most important.
            </div>
            """,
            unsafe_allow_html=True,
            )


    st.markdown(
        """
        <div style="font-size:12px; color:gray; text-align:center;">
        <b>Credits:</b><br>
        • Source dataset: Kaggle | Comprehensive Diabetes Clinical Dataset (100k rows) | Author: Priyam Choksi | MIT License<br>
        • Model: Random Forest Classifier trained on age, BMI, HbA1c, blood glucose, hypertension, heart disease and gender<br>
        • Explainability: SHAP (Lundberg & Lee) for feature impact analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )
