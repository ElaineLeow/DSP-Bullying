import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================================
# 1. PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="EduSafe | Bullying Risk Prediction",
    layout="wide",
    page_icon="🛡️"
)

# =========================================
# 2. LOAD NEW MODEL ASSETS
# =========================================
@st.cache_resource
def load_assets():
    return joblib.load("bullying_risk_model_xgboost.pkl")  # <-- updated

assets = load_assets()
model = assets["model"]
scaler = assets["scaler"]
features = assets["features"]
threshold = assets["threshold"]  # <-- this is the threshold optimized during training

# =========================================
# 3. COLUMN MAPPING (CSV/UI → MODEL)
# =========================================
COLUMN_MAP = {
    "class_size": "class_size",
    "Loneliness": "how_often_do_you_feel_lonely_at_school?",
    "Stress": "how_stressed_do_you_feel_at_school",
    "Safety": "how_safe_do_you_feel_at_school?",
    "Self_Esteem": "how_would_you_rate_your_self-esteem/confidence?",
    "Sibling_Support": "sibling_support",
    "Peer_Group_Size": "num_close_friends",
    "Family_Support": "how_supportive_is_your_family?",
    "Teacher_Support": "how_supportive_are_your_teachers?"
}

# =========================================
# 4. HELPER FUNCTIONS
# =========================================
def get_category(prob, threshold):
    if prob >= threshold:
        return "High"
    elif prob >= 0.35:
        return "Moderate"
    else:
        return "Low"

# =========================================
# 5. SIDEBAR NAVIGATION
# =========================================
st.sidebar.title("🛡️ EduSafe Dashboard")
mode = st.sidebar.radio(
    "Select Mode",
    ["Individual Screening", "Classroom CSV Upload"]
)

# =========================================
# 6. INDIVIDUAL SCREENING
# =========================================
if mode == "Individual Screening":
    st.title("👤 Individual Student Risk Assessment")

    with st.form("student_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Student Name", "John Doe")
            matric = st.text_input("Matric Number", "MS123456")
        with col2:
            assessment_date = st.date_input("Assessment Date", datetime.now())

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("🚩 Risk Factors")
            loneliness = st.slider("Loneliness", 1, 5, 3)
            stress = st.slider("Stress Level", 1, 5, 3)
        with c2:
            st.subheader("🟢 Support Factors")
            family = st.slider("Family Support", 1, 5, 3)
            teacher = st.slider("Teacher Support", 1, 5, 3)
            sibling = st.slider("Sibling Support", 1, 5, 3)
        with c3:
            st.subheader("🏫 Environment")
            safety = st.slider("School Safety", 1, 5, 3)
            friends = st.slider("Number of Close Friends", 1, 5, 3)
            esteem = st.slider("Self-Esteem", 1, 5, 3)
            class_size = st.number_input("Class Size", min_value=1, max_value=100, value=30)

        submit = st.form_submit_button("Predict Risk")

    if submit:
        # ---- Build ML input
        input_data = {
            "class_size": class_size,
            "how_safe_do_you_feel_at_school?": safety,
            "how_stressed_do_you_feel_at_school": stress,
            "how_often_do_you_feel_lonely_at_school?": loneliness,
            "how_would_you_rate_your_self-esteem/confidence?": esteem,
            "sibling_support": sibling,
            "num_close_friends": friends,
            "how_supportive_is_your_family?": family,
            "how_supportive_are_your_teachers?": teacher
        }
        X_input = pd.DataFrame([input_data])[features]
        X_scaled = scaler.transform(X_input)

        prob = model.predict_proba(X_scaled)[:, 1][0]
        risk_score = prob * 100
        category = get_category(prob, threshold)  # <-- uses optimized threshold from training

        st.divider()
        # --- Gauge + Spider chart
        colA, colB = st.columns(2)
        with colA:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": f"Risk Index: {name}"},
                gauge={"axis": {"range": [0, 100]},
                       "steps":[{"range":[0,35],"color":"#00c853"},
                                {"range":[35,70],"color":"#ffbb33"},
                                {"range":[70,100],"color":"#ff4b4b"}] }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with colB:
            categories = ["Loneliness", "Stress", "Family", "Teacher", "Safety", "Self-Esteem"]
            values = [loneliness, stress, family, teacher, safety, esteem]
            fig_spider = go.Figure(go.Scatterpolar(
                r=values, theta=categories, fill='toself', line_color='green'
            ))
            fig_spider.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), showlegend=False, height=400)
            st.plotly_chart(fig_spider, use_container_width=True)

        st.divider()
        # --- Key Insights
        st.subheader("📌 Key Insights")
        insights = []
        if loneliness >= 4: insights.append("• **Isolation:** High loneliness detected.")
        if stress >= 4: insights.append("• **Stress:** Significant emotional pressure.")
        if safety <= 2: insights.append("• **Environment:** Student feels unsafe at school.")
        if teacher <= 2: insights.append("• **Staff Connection:** Weak bond with teachers.")
        if not insights:
            st.info("Factors are currently within stable ranges.")
        else:
            for item in insights:
                st.info(item)

        # --- Action Plan
        st.subheader("📋 Recommended Action")
        if category=="High":
            st.error("🚨 Immediate Referral: Contact counselor and parents.")
        elif category=="Moderate":
            st.warning("⚠️ Monitor & provide peer support.")
        else:
            st.success("✅ Stable: No immediate action required.")

        # --- Download Report
        report = f"""EDUSAFE REPORT
Date: {assessment_date}
Student: {name} | Matric: {matric}
Risk Score: {risk_score:.1f}%
Category: {category}

Insights:
{chr(10).join(insights) if insights else 'Stable'}

Recommendation:
{"Immediate Referral" if category=='High' else "Monitor & Peer Support" if category=='Moderate' else "No Action Required"}
"""
        st.download_button("📩 Download Report", report, file_name=f"EduSafe_{matric}.txt")

# =========================================
# 7. CLASSROOM CSV UPLOAD
# =========================================
else:
    st.title("📊 Classroom Risk Analysis")
    template = pd.DataFrame(columns=features)
    st.download_button("📥 Download CSV Template", template.to_csv(index=False), "edu_template.csv")

    uploaded_file = st.file_uploader("Upload Classroom CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Ensure all required columns exist
        missing = set(features) - set(df.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X_input = df[features]
        X_scaled = scaler.transform(X_input)

        df["Risk_Probability"] = model.predict_proba(X_scaled)[:,1]
        df["Risk_Score (%)"] = df["Risk_Probability"]*100
        df["Category"] = df["Risk_Probability"].apply(lambda x: get_category(x, threshold))

        # --- Class Composition Pie Chart
        st.subheader("📌 Class Composition")
        fig_pie = px.pie(df, names='Category', color='Category',
                         color_discrete_map={'Low':'#00c853','Moderate':'#ffbb33','High':'#ff4b4b'})
        st.plotly_chart(fig_pie)

        # --- Average Class Vulnerability (Spider)
        st.subheader("📊 Average Class Vulnerability")
        avg_vals = [
            df["how_often_do_you_feel_lonely_at_school?"].mean(),
            df["how_stressed_do_you_feel_at_school"].mean(),
            df["how_supportive_is_your_family?"].mean(),
            df["how_supportive_are_your_teachers?"].mean(),
            df["how_safe_do_you_feel_at_school?"].mean(),
            df["how_would_you_rate_your_self-esteem/confidence?"].mean()
        ]
        fig_spider_avg = go.Figure(go.Scatterpolar(
            r=avg_vals, theta=['Loneliness','Stress','Family Sup','Teacher Sup','Safety','Self-Esteem'],
            fill='toself', line_color='green'
        ))
        fig_spider_avg.update_layout(polar=dict(radialaxis=dict(range=[0,5])), showlegend=False, height=400)
        st.plotly_chart(fig_spider_avg)

        # --- Top 5 High Risk Students
        st.subheader("🚨 Priority Intervention List (Top 5)")
        if "Student_Name" in df.columns and "Matric_Number" in df.columns:
            st.dataframe(
                df.sort_values("Risk_Score (%)", ascending=False)
                .head(5)[["Student_Name", "Matric_Number", "Risk_Score (%)", "Category"]]
            )
        else:
            # fallback if no name/matric in CSV
            st.dataframe(
                df.sort_values("Risk_Score (%)", ascending=False)
                .head(5)[["Risk_Score (%)", "Category"]]
            )

        # --- Summary Metrics
        st.subheader("📉 Institutional Insights & Recommendations")
        avg_risk = df['Risk_Score (%)'].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Class Risk", f"{avg_risk:.1f}%")
        col2.metric("High Risk Students", (df["Category"]=="High").sum())
        col3.metric("Moderate Risk Students", (df["Category"]=="Moderate").sum())

        if avg_risk>50:
            st.error("💡 High overall risk: plan class-wide mental health workshop.")
        elif (df["Category"]=="High").sum()>3:
            st.warning("💡 Multiple high-risk students: recommend group counseling.")
        else:
            st.success("💡 Class environment is stable.")
