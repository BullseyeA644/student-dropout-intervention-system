import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# =========================
# LOAD ASSETS
# =========================
@st.cache_resource
def load_assets():
    model = load_model("student_success_binary_model.h5")
    preprocessor = joblib.load("preprocessor.pkl")
    try:
        feature_names = joblib.load("feature_names.pkl")
    except Exception:
        feature_names = None
    return model, preprocessor, feature_names

model, preprocessor, feature_names = load_assets()
def profile_card(title, value, bgcolor="#eef2ff"):
    st.markdown(
        f"""
        <div style="
            background-color: {bgcolor};
            padding: 16px;
            border-radius: 14px;
            border: 1px solid #e5e7eb;
            text-align: center;
        ">
            <div style="
                font-size: 13px;
                color: #374151;   /* dark gray */
                margin-bottom: 6px;
            ">
                {title}
            </div>
            <div style="
                font-size: 18px;
                font-weight: 600;
                color: #111827;   /* almost black */
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Student Dropout Risk Predictor", layout="wide")
st.title("🎓 Student Dropout Risk Prediction & Early Intervention System")
st.write("An AI-based early warning system to identify students at risk of dropping out and suggest intervention actions.")
st.markdown("---")

# =========================
# MAPS
# =========================
gender_map = {"Female": 0, "Male": 1}
yes_no_map = {"No": 0, "Yes": 1}

marital_map = {
    "Single": 1, "Married": 2, "Widower": 3,
    "Divorced": 4, "Common-law marriage": 5, "Legally separated": 6
}

course_map = {
    "Biofuel Production Technologies": 1,
    "Animation and Multimedia Design": 2,
    "Social Service (evening)": 3,
    "Agronomy": 4,
    "Communication Design": 5,
    "Veterinary Nursing": 6,
    "Informatics Engineering": 7,
    "Equiniculture": 8,
    "Management": 9,
    "Social Service": 10,
    "Tourism": 11,
    "Nursing": 12,
    "Oral Hygiene": 13,
    "Advertising and Marketing Management": 14,
    "Journalism and Communication": 15,
    "Basic Education": 16,
    "Management (evening)": 17
}

qualification_map = {
    "Secondary education": 1,
    "Bachelor's degree": 2,
    "Degree": 3,
    "Master's degree": 4,
    "Doctorate": 5,
    "Technical specialization": 14,
    "Professional technical course": 16
}

# =========================
# HELPERS
# =========================
def build_input_dict(
    gender, age, marital, course,
    scholarship, tuition, debtor,
    mother_qual, father_qual,
    enrolled, evaluated, approved, credited,
    app_order
):
    return {
        "Gender": gender,
        "Marital status": marital,
        "Course": course,
        "Scholarship holder": scholarship,
        "Tuition fees up to date": tuition,
        "Debtor": debtor,
        "Mother's qualification": mother_qual,
        "Father's qualification": father_qual,
        "Age at enrollment": age,
        "Curricular units 1st sem (enrolled)": enrolled,
        "Curricular units 1st sem (evaluations)": evaluated,
        "Curricular units 1st sem (approved)": approved,
        "Curricular units 1st sem (credited)": credited,
        "Application order": app_order
    }

def predict_success_probability(input_dict):
    df = pd.DataFrame([input_dict])
    processed = preprocessor.transform(df)
    prob = model.predict(processed, verbose=0)[0][0]
    return float(prob)

def format_pct(x):
    return f"{x*100:.1f}%"

def risk_bucket(success_prob):
    # success_prob is P(success). dropout risk = 1 - success_prob
    if success_prob < 0.40:
        return "High Risk", "🚨"
    elif success_prob < 0.70:
        return "Medium Risk", "⚠️"
    else:
        return "Low Risk", "✅"

def risk_score_0_to_10(success_prob):
    # higher score = higher risk
    return round((1.0 - success_prob) * 10, 1)

def confidence_indicator(success_prob):
    # simple: distance from 0.5
    return round(abs(success_prob - 0.5) * 2, 2)  # 0..1

def student_risk_profile(base_input):
    enrolled = base_input["Curricular units 1st sem (enrolled)"]
    evaluated = base_input["Curricular units 1st sem (evaluations)"]
    approved = base_input["Curricular units 1st sem (approved)"]

    pass_rate = (approved / evaluated) if evaluated > 0 else 0
    engagement_rate = (evaluated / enrolled) if enrolled > 0 else 0

    fee_ok = base_input["Tuition fees up to date"] == 1
    debtor = base_input["Debtor"] == 1

    academic_risk = (pass_rate < 0.6) or (engagement_rate < 0.75)
    financial_risk = (not fee_ok) or debtor

    if academic_risk and financial_risk:
        return "Academically & Financially At-Risk", "🚨"
    elif academic_risk:
        return "Academically At-Risk", "📚"
    elif financial_risk:
        return "Financially At-Risk", "💰"
    else:
        return "High Performer", "🌟"

def simulate_change(base_input, key, new_value):
    mod = base_input.copy()
    mod[key] = new_value
    return predict_success_probability(mod)

def compute_rates(base_input):
    enrolled = base_input["Curricular units 1st sem (enrolled)"]
    evaluated = base_input["Curricular units 1st sem (evaluations)"]
    approved = base_input["Curricular units 1st sem (approved)"]

    pass_rate = (approved / evaluated) if evaluated > 0 else 0.0
    engagement_rate = (evaluated / enrolled) if enrolled > 0 else 0.0
    return pass_rate, engagement_rate

def avg_student_baselines():
    """
    Lightweight, model-free baselines (no dataset loaded here).
    These are conservative heuristics for comparison.

    If you want true averages from training data later,
    we can save them from Colab and load here.
    """
    return {
        "pass_rate_avg": 0.65,
        "engagement_avg": 0.80,
        "tuition_up_to_date_avg": 0.75
    }

def goal_based_plan(base_input, base_prob, target_success_prob):
    """
    Find minimal changes to reach target success probability.
    Returns a dict with plan details.
    """
    plan = {
        "achieved": False,
        "steps": [],
        "final_prob": base_prob,
        "notes": []
    }

    # Step candidates: fees up to date, remove debtor, add scholarship, increase approvals
    # We'll try single actions first, then combinations.
    candidates = []

    # Single actions
    if base_input["Tuition fees up to date"] == 0:
        p = simulate_change(base_input, "Tuition fees up to date", 1)
        candidates.append(("Clear tuition fees", {"Tuition fees up to date": 1}, p))

    if base_input["Debtor"] == 1:
        p = simulate_change(base_input, "Debtor", 0)
        candidates.append(("Resolve debtor status", {"Debtor": 0}, p))

    if base_input["Scholarship holder"] == 0:
        p = simulate_change(base_input, "Scholarship holder", 1)
        candidates.append(("Obtain scholarship support", {"Scholarship holder": 1}, p))

    # Academic improvements: +k approvals
    approved = int(base_input["Curricular units 1st sem (approved)"])
    evaluated = int(base_input["Curricular units 1st sem (evaluations)"])
    max_add = max(0, evaluated - approved)

    for k in range(1, min(6, max_add + 1)):
        mod = base_input.copy()
        mod["Curricular units 1st sem (approved)"] = approved + k
        p = predict_success_probability(mod)
        candidates.append((f"Pass {k} additional subject(s)", {"Curricular units 1st sem (approved)": approved + k}, p))

    # Rank single candidates by probability achieved (descending)
    candidates_sorted = sorted(candidates, key=lambda x: x[2], reverse=True)

    # Check if any single action reaches the goal
    for name, changes, p in candidates_sorted:
        if p >= target_success_prob:
            plan["achieved"] = True
            plan["steps"].append((name, changes))
            plan["final_prob"] = p
            return plan

    # If not achieved, try small combinations: top 6 actions pairwise (greedy-ish)
    top = candidates_sorted[:6]
    best_combo = None

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            name1, ch1, _ = top[i]
            name2, ch2, _ = top[j]

            mod = base_input.copy()
            for k, v in ch1.items():
                mod[k] = v
            for k, v in ch2.items():
                # keep academic validity if approvals > evaluations
                mod[k] = v

            # Validate academic constraints if approvals changed
            if mod["Curricular units 1st sem (approved)"] > mod["Curricular units 1st sem (evaluations)"]:
                continue

            p = predict_success_probability(mod)
            if (best_combo is None) or (p > best_combo[2]):
                best_combo = (f"{name1} + {name2}", {**ch1, **ch2}, p)

    if best_combo and best_combo[2] >= target_success_prob:
        plan["achieved"] = True
        plan["steps"].append((best_combo[0], best_combo[1]))
        plan["final_prob"] = best_combo[2]
    else:
        # Not achieved: provide best effort suggestion
        if best_combo:
            plan["notes"].append("Target not fully achieved with simple interventions; showing best available improvement.")
            plan["steps"].append((best_combo[0], best_combo[1]))
            plan["final_prob"] = best_combo[2]
        else:
            plan["notes"].append("No valid intervention combination found under current constraints.")

    return plan

def intervention_letter(base_input, success_prob):
    pass_rate, engagement_rate = compute_rates(base_input)
    fee_ok = base_input["Tuition fees up to date"] == 1
    debtor = base_input["Debtor"] == 1
    scholar = base_input["Scholarship holder"] == 1
    risk_level, _ = risk_bucket(success_prob)
    profile_label, _ = student_risk_profile(base_input)

    letter = (
        f"This student is currently assessed as **{risk_level}** with an estimated **{format_pct(1-success_prob)} dropout risk** "
        f"and **{format_pct(success_prob)} probability of successful continuation**. The risk profile is **{profile_label}**. "
        f"Academic indicators show an engagement rate of approximately **{engagement_rate*100:.0f}%** and a pass rate of **{pass_rate*100:.0f}%**. "
        f"Financial indicators show tuition is **{'up to date' if fee_ok else 'not up to date'}** and the student is **{'a debtor' if debtor else 'not a debtor'}**. "
        f"{'Scholarship support is present.' if scholar else 'No scholarship support is indicated.'} "
        f"Recommended interventions include targeted academic mentoring to improve pass rate, proactive attendance/engagement follow-up, "
        f"and financial counseling or fee clearance support where applicable. A follow-up review is recommended after the next assessment cycle."
    )
    return letter

# =========================
# SIDEBAR FORM
# =========================
with st.sidebar.form("student_form"):
    st.header("📋 Student Profile")

    gender_label = st.selectbox("Gender", list(gender_map.keys()))
    gender = gender_map[gender_label]

    age = st.slider("Age at enrollment", 16, 60, 20)

    marital_label = st.selectbox("Marital Status", list(marital_map.keys()))
    marital = marital_map[marital_label]

    course_label = st.selectbox("Course", list(course_map.keys()))
    course = course_map[course_label]

    scholarship_label = st.selectbox("Scholarship Holder", list(yes_no_map.keys()))
    scholarship = yes_no_map[scholarship_label]

    tuition_label = st.selectbox("Tuition Fees Up To Date", list(yes_no_map.keys()))
    tuition = yes_no_map[tuition_label]

    debtor_label = st.selectbox("Debtor", list(yes_no_map.keys()))
    debtor = yes_no_map[debtor_label]

    mother_qual_label = st.selectbox("Mother's Education Level", list(qualification_map.keys()))
    mother_qual = qualification_map[mother_qual_label]

    father_qual_label = st.selectbox("Father's Education Level", list(qualification_map.keys()))
    father_qual = qualification_map[father_qual_label]

    enrolled = st.number_input("Subjects Registered", 0, 20, 6)
    evaluated = st.number_input("Subjects Attempted", 0, enrolled, min(6, enrolled))
    approved = st.number_input("Subjects Passed", 0, evaluated, min(5, evaluated))
    credited = st.number_input("Subjects Credited (Previous Studies)", 0, 20, 0)

    app_order = st.slider("Course Preference Rank (1 = First Choice)", 1, 10, 1)

    submit = st.form_submit_button("🔍 Predict & Analyze")

# =========================
# SESSION STATE INIT
# =========================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# =========================
# PREDICTION
# =========================
if submit:
    if approved > evaluated or evaluated > enrolled:
        st.error("Invalid values: Passed ≤ Attempted ≤ Registered.")
        st.session_state.predicted = False
    else:
        base_input = build_input_dict(
            gender, age, marital, course,
            scholarship, tuition, debtor,
            mother_qual, father_qual,
            enrolled, evaluated, approved, credited,
            app_order
        )
        base_prob = predict_success_probability(base_input)

        st.session_state.predicted = True
        st.session_state.base_input = base_input
        st.session_state.base_prob = base_prob

# =========================
# OUTPUT
# =========================
if st.session_state.predicted:
    base_input = st.session_state.base_input
    base_prob = st.session_state.base_prob

    risk_level, risk_icon = risk_bucket(base_prob)
    rscore = risk_score_0_to_10(base_prob)
    conf = confidence_indicator(base_prob)
    profile_label, profile_icon = student_risk_profile(base_input)

    pass_rate, engagement_rate = compute_rates(base_input)

    # =========================
    # TOP SUMMARY CARD
    # =========================
    with st.container(border=True):
        st.subheader("📊 Prediction Summary")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Probability of Success", format_pct(base_prob))
        colB.metric("Dropout Risk", format_pct(1 - base_prob))
        colC.metric("Risk Score (0–10)", f"{rscore}")
        colD.metric("Confidence", f"{conf}")

        st.progress(float(base_prob))

        if risk_level == "High Risk":
            st.error(f"{risk_icon} **High Risk of Dropout**")
        elif risk_level == "Medium Risk":
            st.warning(f"{risk_icon} **Medium Risk — Needs Attention**")
        else:
            st.success(f"{risk_icon} **Low Risk — Likely to Continue Successfully**")

        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Student Risk Profile:** {profile_icon} **{profile_label}**")

    st.markdown("---")

    # =========================
    # AUTO SUMMARY REPORT
    # =========================
    with st.container(border=True):
        st.subheader("📝 Auto Summary Report")

        fee_ok = base_input["Tuition fees up to date"] == 1
        is_debtor = base_input["Debtor"] == 1
        is_scholar = base_input["Scholarship holder"] == 1

        summary = (
            f"The model predicts a **{format_pct(base_prob)} probability of success** "
            f"(dropout risk **{format_pct(1-base_prob)}**), categorized as **{risk_level}**. "
            f"Academic engagement is approximately **{engagement_rate*100:.0f}%** (attempted vs registered), "
            f"with a pass rate of **{pass_rate*100:.0f}%** (passed vs attempted). "
            f"Financial indicators show tuition is {'up to date' if fee_ok else 'not up to date'} and the student is "
            f"{'a debtor' if is_debtor else 'not a debtor'}. "
            f"{'Scholarship support is present.' if is_scholar else 'No scholarship support is indicated.'}"
        )
        st.info(summary)

    # =========================
    # RECOMMENDATIONS
    # =========================
    st.markdown("---")
    with st.container(border=True):
        st.subheader("💡 Personalized Recommendations")

        recs = []

        # Academic
        if base_input["Curricular units 1st sem (evaluations)"] > 0 and pass_rate < 0.6:
            recs.append("📚 Improve pass rate: target **≥ 60%** passes in attempted subjects (study plan + mentoring).")
        if base_input["Curricular units 1st sem (enrolled)"] > 0 and engagement_rate < 0.75:
            recs.append("🧠 Increase engagement: attempt more registered subjects (attendance + assignment completion).")
        if base_input["Curricular units 1st sem (approved)"] < max(2, int(0.5 * base_input["Curricular units 1st sem (evaluations)"])):
            recs.append("✅ Focus on clearing backlogs early; prioritize high-credit or prerequisite courses first.")

        # Financial
        if base_input["Tuition fees up to date"] == 0:
            recs.append("💳 Clear tuition dues: fee status can significantly improve predicted success.")
        if base_input["Debtor"] == 1:
            recs.append("💰 Debtor flag detected: consider instalments, financial counseling, or institutional support.")
        if base_input["Scholarship holder"] == 0:
            recs.append("🎓 Explore scholarship options: financial aid may reduce stress and improve outcomes.")

        # If none
        if not recs:
            recs.append("🌟 Keep up consistent performance and maintain financial/academic stability.")

        for r in recs:
            st.write(f"- {r}")

    # =========================
    # GOAL-BASED SIMULATION
    # =========================
    st.markdown("---")
    with st.container(border=True):
        st.subheader("🎯 Goal-Based Intervention Planner")
        st.caption("Set a target dropout risk. The system suggests minimal actions to reach that target (based on simulation).")

        target_risk_pct = st.slider("Target Dropout Risk (%)", 5, 80, 30)
        target_success_prob = 1 - (target_risk_pct / 100)

        plan = goal_based_plan(base_input, base_prob, target_success_prob)

        st.write(f"Current dropout risk: **{format_pct(1-base_prob)}**")
        st.write(f"Target dropout risk: **{target_risk_pct:.0f}%**")

        if plan["achieved"]:
            st.success(f"✅ Target is achievable. Estimated success probability after plan: **{format_pct(plan['final_prob'])}**")
        else:
            st.warning(f"⚠️ Target may not be fully achievable with simple interventions. Best estimated success probability: **{format_pct(plan['final_prob'])}**")

        if plan["steps"]:
            st.markdown("**Recommended Plan:**")
            for step_name, changes in plan["steps"]:
                st.write(f"- **{step_name}**")
                for k, v in changes.items():
                    st.write(f"  - Set `{k}` → **{v}**")

        for note in plan["notes"]:
            st.info(note)

    # =========================
    # MULTI-PARAMETER WHAT-IF SIMULATOR
    # =========================
    st.markdown("---")
    with st.container(border=True):
        st.subheader("🔮 What-If Simulator")
        st.caption("Adjust multiple inputs to see how predicted probability changes (no retraining).")

        col1, col2, col3 = st.columns(3)

        with col1:
            wf_approved = st.slider(
                "Subjects Passed",
                0,
                int(base_input["Curricular units 1st sem (enrolled)"]),
                int(base_input["Curricular units 1st sem (approved)"])
            )
            wf_evaluated = st.slider(
                "Subjects Attempted",
                0,
                int(base_input["Curricular units 1st sem (enrolled)"]),
                int(base_input["Curricular units 1st sem (evaluations)"])
            )

        with col2:
            wf_tuition = st.selectbox(
                "Tuition Fees Up To Date",
                ["No", "Yes"],
                index=int(base_input["Tuition fees up to date"])
            )
            wf_debtor = st.selectbox(
                "Debtor",
                ["No", "Yes"],
                index=int(base_input["Debtor"])
            )

        with col3:
            wf_scholarship = st.selectbox(
                "Scholarship Holder",
                ["No", "Yes"],
                index=int(base_input["Scholarship holder"])
            )
            wf_credited = st.slider(
                "Credited Subjects (Previous Studies)",
                0,
                20,
                int(base_input["Curricular units 1st sem (credited)"])
            )

        if wf_approved > wf_evaluated:
            st.error("What-if invalid: Passed ≤ Attempted must hold.")
        else:
            wf_input = base_input.copy()
            wf_input["Curricular units 1st sem (approved)"] = int(wf_approved)
            wf_input["Curricular units 1st sem (evaluations)"] = int(wf_evaluated)
            wf_input["Tuition fees up to date"] = yes_no_map[wf_tuition]
            wf_input["Debtor"] = yes_no_map[wf_debtor]
            wf_input["Scholarship holder"] = yes_no_map[wf_scholarship]
            wf_input["Curricular units 1st sem (credited)"] = int(wf_credited)

            wf_prob = predict_success_probability(wf_input)
            delta = wf_prob - base_prob

            cX, cY, cZ = st.columns(3)
            cX.metric("New Probability of Success", format_pct(wf_prob), delta=f"{delta*100:+.1f}%")
            cY.metric("New Dropout Risk", format_pct(1 - wf_prob), delta=f"{((1-wf_prob)-(1-base_prob))*100:+.1f}%")
            new_profile, new_icon = student_risk_profile(wf_input)
            with cZ:
             profile_card("New Risk Profile", f"{new_icon} {new_profile}")


    # =========================
    # RANKED ACTION LIST
    # =========================
    st.markdown("---")
    with st.container(border=True):
        st.subheader("📌 Ranked Action Plan (Most Impactful First)")
        st.caption("Estimated impact of common interventions on success probability for this student profile.")

        actions = []

        # Financial actions
        if base_input["Tuition fees up to date"] == 0:
            p = simulate_change(base_input, "Tuition fees up to date", 1)
            actions.append(("Clear tuition fees (mark up to date)", p - base_prob))

        if base_input["Debtor"] == 1:
            p = simulate_change(base_input, "Debtor", 0)
            actions.append(("Resolve debtor status", p - base_prob))

        if base_input["Scholarship holder"] == 0:
            p = simulate_change(base_input, "Scholarship holder", 1)
            actions.append(("Obtain scholarship support", p - base_prob))

        # Academic action: pass +1 subject if possible
        if base_input["Curricular units 1st sem (approved)"] < base_input["Curricular units 1st sem (evaluations)"]:
            p = simulate_change(
                base_input,
                "Curricular units 1st sem (approved)",
                int(base_input["Curricular units 1st sem (approved)"] + 1)
            )
            actions.append(("Pass one additional subject", p - base_prob))

        if not actions:
            st.write("No obvious single-step improvements detected. Maintain performance and monitor regularly.")
        else:
            actions = sorted(actions, key=lambda x: x[1], reverse=True)
            for i, (name, impact) in enumerate(actions, start=1):
                st.write(f"**{i}. {name}** → estimated success change: **{impact*100:+.1f}%**")

    # =========================
    # AVERAGE STUDENT COMPARISON
    # =========================
    st.markdown("---")
    with st.container(border=True):
        st.subheader("📊 Comparison with Average Student (Baseline)")
        st.caption("Baseline comparison uses conservative heuristics. (We can replace with true dataset averages later.)")

        baselines = avg_student_baselines()
        pass_avg = baselines["pass_rate_avg"]
        eng_avg = baselines["engagement_avg"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Your Pass Rate", f"{pass_rate*100:.0f}%", delta=f"{(pass_rate-pass_avg)*100:+.0f}% vs avg")
        col2.metric("Your Engagement", f"{engagement_rate*100:.0f}%", delta=f"{(engagement_rate-eng_avg)*100:+.0f}% vs avg")

        tuition_up = 1 if base_input["Tuition fees up to date"] == 1 else 0
        tuition_avg = baselines["tuition_up_to_date_avg"]
        col3.metric("Tuition Up To Date", "Yes" if tuition_up else "No", delta=f"{(tuition_up - tuition_avg)*100:+.0f}% vs avg")

        st.info(
            "Baseline values are indicative, assumed cohort-level references provided for contextual interpretation. "
            "They are not calculated from the training dataset and do not influence model predictions." 
        )

    # =========================
    # AUTO-GENERATED INTERVENTION LETTER
    # =========================
    st.markdown("---")
    with st.container(border=True):
        
        note = intervention_letter(base_input, base_prob)
        st.subheader("📄 Intervention Summary (Readable View)")

        st.markdown(f"""
        ### 🔎 Risk Assessment
        - **Risk Level:** {risk_level}
        - **Dropout Risk:** {format_pct(1-base_prob)}
        - **Probability of Success:** {format_pct(base_prob)}
        - **Risk Profile:** {profile_label}

        ### 📚 Academic Indicators
        - **Engagement Rate:** {engagement_rate*100:.0f}%
        - **Pass Rate:** {pass_rate*100:.0f}%

        ### 💰 Financial Indicators
        - **Tuition Fees Up To Date:** {"Yes" if fee_ok else "No"}
        - **Debtor Status:** {"Yes" if is_debtor else "No"}
        - **Scholarship Holder:** {"Yes" if is_scholar else "No"}

        ### 💡 Recommended Interventions
        - Targeted academic mentoring to improve pass rate
        - Proactive engagement and attendance monitoring
        - Financial counseling or tuition fee clearance support (if applicable)

        📌 **Follow-up review recommended after the next assessment cycle.**
        """)


    # =========================
    # DEBUG / INFO
    # =========================
    with st.expander("🔧 Debug / Model Info"):
        st.write("Model and preprocessor loaded successfully.")
        st.write(f"Success probability (model output): {base_prob:.4f}")
        if feature_names is None:
            st.write("feature_names.pkl not loaded (optional).")
        else:
            st.write(f"feature_names available: {len(feature_names)} encoded features.")
