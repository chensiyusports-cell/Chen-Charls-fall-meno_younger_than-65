# ==================== 0. 导入库 ====================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
# ==================== 1. 基础配置 ====================
# 加载训练好的随机森林模型（确保 RF.pkl 与脚本同目录）
model = joblib.load("xgb_model.pkl")

# 加载测试数据（用于 LIME 解释，确保 X_test.csv 与脚本同目录）
X_test = pd.read_csv("traindata_for_before65.csv")

# 特征名称（需与你训练模型时的特征顺序一致）
feature_names = [
    'time_5_sts',
    'Waist_Circumference',
    'body_mass',
    'CESD10',
    'Height',
    'unDomain_2KG',
    'Fallen_down_history',
    'Pulse',
    'pef_mean',
    'Age',
    'always_bothered_by_pain',
    'self_rated_health1',
    'satisfaction_life_overall',
    'PP'
]

# ==================== 2. Streamlit 页面配置 ====================
st.set_page_config(page_title="Fall Risk Prediction", layout="wide")
st.title("Fall Risk Prediction Model for Chinese Postmenopausal Women Aged < 65")
st.markdown("Please complete the following information to obtain an individualized fall risk estimate.")

# ==================== 3. 特征输入组件（按编码规则设计） ====================
time_5_sts = st.number_input(
    "**5-times Sit-to-Stand Test Time (s)**",
    min_value=0.0,
    step=0.1
)

Waist_Circumference = st.number_input(
    "**Waist Circumference (cm)**",
    min_value=0.0,
    step=0.1
)

body_mass = st.number_input(
    "**Body Weight (kg)**",
    min_value=0.0,
    step=0.1
)

# ===== CES-D 10: 自动计算（0–30）=====
st.markdown("### CES-D 10")

CESD_OPTIONS = [0, 1, 2, 3]
CESD_LABELS = {
    0: "Rarely or none of the time (<1 day)",
    1: "Some or a little of the time (1–2 days)",
    2: "Occasionally or a moderate amount of the time (3–4 days)",
    3: "Most or all of the time (5–7 days)"
}

def cesd_item(question: str, key: str) -> int:
    return st.selectbox(
        question,
        options=CESD_OPTIONS,
        format_func=lambda x: CESD_LABELS[x],
        key=key
    )

# 8 个负向题（直接加分）
felt_depressed1 = cesd_item("How often did you feel depressed during the past week?", "felt_depressed1")
everything_was_an_effort = cesd_item("How often did you feel that everything was an effort during the past week?", "everything_was_an_effort")
felt_fearful = cesd_item("How often did you feel fearful during the past week?", "felt_fearful")
poor_sleep = cesd_item("How often did you have restless sleep during the past week?", "poor_sleep")
felt_lonely = cesd_item("How often did you feel lonely during the past week?", "felt_lonely")
felt_could_not_go_on_with_my_life = cesd_item("How often did you feel you could not go on with your life during the past week?", "felt_could_not_go_on_with_my_life")
bothered_by_trivial_things = cesd_item("How often were you bothered by things that don't usually bother you during the past week?", "bothered_by_trivial_things")
hard_to_concentrate = cesd_item("How often did you have trouble keeping your mind on what you were doing during the past week?", "hard_to_concentrate")

# 2 个正向题（需要反向计分：3-x）
felt_happy = cesd_item("How often did you feel happy during the past week?", "felt_happy")
hopeful_about_future = cesd_item("How often did you feel hopeful about the future during the past week?", "hopeful_about_future")

CESD10 = (
    felt_depressed1
    + everything_was_an_effort
    + felt_fearful
    + poor_sleep
    + felt_lonely
    + felt_could_not_go_on_with_my_life
    + bothered_by_trivial_things
    + hard_to_concentrate
    + (3 - felt_happy)
    + (3 - hopeful_about_future)
)

st.number_input(
    "CES-D 10 Total Score (0–30)",
    min_value=0,
    max_value=30,
    step=1,
    value=int(CESD10),
    disabled=True
)

Height = st.number_input(
    "**Height (cm)**",
    min_value=0.0,
    step=0.1
)

unDomain_2KG = st.number_input(
    "**Maximum Non-dominant Arm Biceps Curl Repetitions with 2 kg Load**",
    min_value=0,
    step=1
)

Fallen_down_history = st.selectbox(
    "**Have you fallen down before?**",
    options=[0, 1],
    format_func=lambda x: "yes" if x == 1 else "no"
)


Pulse = st.number_input(
    "**Resting Heart Rate (bpm)**",
    min_value=0.0,
    step=0.1
)

pef_mean = st.number_input(
    "**Peak Expiratory Flow (L/min)**",
    min_value=0.0,
    step=0.1
)

Age = st.number_input(
    "**Age (years)**",
    min_value=0,
    step=1
)

always_bothered_by_pain = st.selectbox(
    "**Are you always bothered by pain?**",
    options=[1, 2],
    format_func=lambda x: "yes" if x == 1 else "no"
)

self_rated_health1 = st.selectbox(
    "**How would you rate your health?**",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "Very good",
        2: "Good",
        3: "Fair",
        4: "Poor",
        5: "Very poor"
    }[x]
)


satisfaction_life_overall = st.selectbox(
    "**How satisfied are you with your life overall?**",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "Very satisfied",
        2: "Satisfied",
        3: "Neutral",
        4: " Dissatisfied",
        5: "Very dissatisfied"
    }[x]
)

PP = st.number_input(
    "**Pulse Pressure (mmHg)**",
    min_value=0.0,
    step=0.1
)


# ==================== 4. 数据处理与预测 ====================
feature_values = [
    time_5_sts,
    Waist_Circumference,
    body_mass,
    CESD10,
    Height,
    unDomain_2KG,
    Fallen_down_history,
    Pulse,
    pef_mean,
    Age,
    always_bothered_by_pain,
    self_rated_health1,
    satisfaction_life_overall,
    PP
]

features = np.array([feature_values])

if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]          # 0: 低风险, 1: 高风险
    predicted_proba = model.predict_proba(features)[0]    # 概率

    # ==================== 5. 结果展示 ====================
    st.subheader("📊Prediction Results")
    st.write(
        f"**Risk Probability：** "
        f"Low risk {predicted_proba[0]:.2%} ｜ High risk {predicted_proba[1]:.2%}"
    )
    st.write(
    "**Note**: This prediction model provides estimated fall risk, with probabilities typically falling within an intermediate range (e.g., 25%–75%). Therefore, values toward the upper end of this range (such as 65%) should be interpreted as relatively elevated risk."
    )

# ==================== 6. LIME 解释（无滚动版本） ====================
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

st.subheader("🔍LIME-based Feature Contribution Analysis")

X_test1 = X_test[feature_names]

lime_explainer = LimeTabularExplainer(
    training_data=X_test1.values,
    feature_names=feature_names,
    class_names=["Low Fall Risk", "High Fall Risk"],
    mode="classification"
)

lime_exp = lime_explainer.explain_instance(
    data_row=features.flatten(),
    predict_fn=model.predict_proba,
    num_features=16
)

lime_html = lime_exp.as_html(show_table=True)
fixed_width = 1400

wrapped_html = f"""
<style>
div[style*="overflow-y"],
div[style*="overflow: auto"] {{
    max-height: 800px !important;
}}
</style>

<div style="width: {fixed_width}px;">
    {lime_html}
</div>
"""

st.components.v1.html(
    wrapped_html,
    height=900,
    scrolling=True
)



    


































