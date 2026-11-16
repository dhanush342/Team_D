"""
AI-Based Industrial Maintenance and Process Optimization System
Interactive Streamlit Dashboard
Infosys Springboard Project Prototype | AI4I 2020 Predictive Maintenance Dataset
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "models" / "failure_model.pkl"
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
REPORTS_DIR = ROOT / "reports"

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial AI | Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial Dark/Modern Theme Styling
st.markdown("""
<style>
    /* Metric Card styling */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Status badge pill */
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pill-green {
        background-color: rgba(16, 185, 129, 0.18);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .pill-amber {
        background-color: rgba(245, 158, 11, 0.18);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .pill-red {
        background-color: rgba(239, 68, 68, 0.18);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Diagnostics Card */
    .diag-box {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-left: 5px solid;
    }
    .diag-box-safe {
        background-color: rgba(16, 185, 129, 0.08);
        border-left-color: #10b981;
    }
    .diag-box-warning {
        background-color: rgba(245, 158, 11, 0.08);
        border-left-color: #f59e0b;
    }
    .diag-box-danger {
        background-color: rgba(239, 68, 68, 0.08);
        border-left-color: #ef4444;
    }

    /* Header subtitle banner */
    .hero-banner {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODEL & DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_bundle(path_str: str):
    """Load pre-trained models bundle from disk."""
    return joblib.load(path_str)


@st.cache_data(show_spinner=False)
def load_metrics_data(path_str: str):
    """Load model evaluation metrics table."""
    p = Path(path_str)
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_dataset_summary(path_str: str):
    """Load JSON summary of the dataset."""
    p = Path(path_str)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_sample_dataset(path_str: str, n_samples: int = 50):
    """Load sample rows from the raw dataset for batch demonstration."""
    p = Path(path_str)
    if p.exists():
        df = pd.read_csv(p)
        return df.sample(min(n_samples, len(df)), random_state=42)
    return None


# Verify that trained model bundle exists
if not BUNDLE_PATH.exists():
    st.error("⚠️ Model file not found (`models/failure_model.pkl`). Please run: `python train_model.py`")
    st.info("Run `python train_model.py` in your terminal to train all 8 models and generate artifacts.")
    st.stop()

bundle = load_model_bundle(str(BUNDLE_PATH))
metrics_df = load_metrics_data(str(METRICS_PATH))
dataset_summary = load_dataset_summary(str(REPORTS_DIR / "dataset_summary.json"))

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: PHYSICS & VISUALIZATIONS
# -----------------------------------------------------------------------------
def calculate_physics_indicators(air_k: float, proc_k: float, rpm: int, torque: float, wear: int):
    """
    Compute domain physical metrics from operational variables:
    - Delta T: Process Temp - Air Temp (Dissipation factor)
    - Mechanical Power: 2 * pi * (rpm/60) * Torque [W]
    - Overstrain Factor: Tool Wear * Torque
    """
    delta_t = round(proc_k - air_k, 2)
    power_watts = round((2 * np.pi * (rpm / 60.0)) * torque, 1)
    overstrain = round(wear * torque, 1)
    return delta_t, power_watts, overstrain


def render_gauge_chart(probability: float, title: str = "Failure Probability"):
    """Render a modern Plotly gauge chart for failure risk."""
    percent = probability * 100.0

    if percent < 30.0:
        bar_color = "#10b981"  # Safe Green
    elif percent < 65.0:
        bar_color = "#f59e0b"  # Warning Amber
    else:
        bar_color = "#ef4444"  # Danger Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 42, "color": "#f8fafc"}},
        title={"text": title, "font": {"size": 16, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b", "ticks": "outside"},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(30, 41, 59, 0.4)",
            "borderwidth": 1,
            "bordercolor": "rgba(148, 163, 184, 0.2)",
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [30, 65], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [65, 100], "color": "rgba(239, 68, 68, 0.15)"}
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 3},
                "thickness": 0.8,
                "value": 65.0
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=25, r=25, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"}
    )
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏭 **Industrial AI Suite**")
    st.caption("AI4I 2020 Predictive Maintenance System")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "⚡ Live Machine Diagnostics",
            "📊 Model Benchmarks & Reports",
            "📁 Batch Fleet Prediction",
            "🔍 Dataset & Leakage Analysis",
            "📘 Architecture & Guidance"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### **System Status**")
    st.markdown("""
    <div style='font-size: 0.85rem; line-height: 1.8;'>
        🟢 <b>Inference Engine:</b> Online<br>
        🤖 <b>Trained Models:</b> 8 Ready<br>
        💾 <b>Dataset:</b> AI4I 2020 (10k records)<br>
        🛡️ <b>Target Leakage:</b> Strictly Mitigated
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Infosys Springboard Project Prototype")


# =============================================================================
# PAGE 1: LIVE MACHINE DIAGNOSTICS
# =============================================================================
if page == "⚡ Live Machine Diagnostics":
    st.title("⚡ Real-Time Machine Diagnostics & Failure Prediction")
    st.markdown("""
    <div class='hero-banner'>
        <strong>Operational Overview:</strong> Input operational sensor telemetry to evaluate machine failure probability, 
        anomaly likelihood, and rule-based physical diagnostic alerts in real time.
    </div>
    """, unsafe_allow_html=True)

    # Preset Telemetry Quick-Loader
    st.markdown("#### ⚡ Quick Telemetry Presets")
    col_pre1, col_pre2, col_pre3, col_pre4, col_pre5 = st.columns(5)
    
    # Initialize session state for input fields if not present
    if "p_type" not in st.session_state:
        st.session_state.p_type = "L"
        st.session_state.p_air = 300.0
        st.session_state.p_proc = 310.0
        st.session_state.p_rpm = 1500
        st.session_state.p_torque = 40.0
        st.session_state.p_wear = 100

    if col_pre1.button("🟢 Normal Operation", use_container_width=True):
        st.session_state.p_type = "L"
        st.session_state.p_air = 298.1
        st.session_state.p_proc = 308.6
        st.session_state.p_rpm = 1550
        st.session_state.p_torque = 38.0
        st.session_state.p_wear = 25

    if col_pre2.button("🔴 Tool Wear Stress", use_container_width=True):
        st.session_state.p_type = "M"
        st.session_state.p_air = 298.5
        st.session_state.p_proc = 309.0
        st.session_state.p_rpm = 1420
        st.session_state.p_torque = 46.0
        st.session_state.p_wear = 238

    if col_pre3.button("🟠 Overheating Risk", use_container_width=True):
        st.session_state.p_type = "L"
        st.session_state.p_air = 303.8
        st.session_state.p_proc = 311.2
        st.session_state.p_rpm = 1380
        st.session_state.p_torque = 54.0
        st.session_state.p_wear = 85

    if col_pre4.button("🔴 Power Overload", use_container_width=True):
        st.session_state.p_type = "H"
        st.session_state.p_air = 299.2
        st.session_state.p_proc = 308.8
        st.session_state.p_rpm = 2750
        st.session_state.p_torque = 68.0
        st.session_state.p_wear = 120

    if col_pre5.button("🟠 Heavy Overstrain", use_container_width=True):
        st.session_state.p_type = "L"
        st.session_state.p_air = 301.0
        st.session_state.p_proc = 310.5
        st.session_state.p_rpm = 1350
        st.session_state.p_torque = 67.0
        st.session_state.p_wear = 215

    st.markdown("---")

    # Input Form Layout
    col_input1, col_input2, col_input3 = st.columns([1.2, 1.2, 1.0])

    with col_input1:
        st.markdown("##### 🌡️ Thermal Parameters")
        air = st.number_input(
            "Air Temperature [K]",
            min_value=295.0, max_value=305.0,
            value=float(st.session_state.p_air), step=0.1,
            help="Ambient workshop temperature (normal operating range: 295.3 - 304.5 K)"
        )
        process = st.number_input(
            "Process Temperature [K]",
            min_value=305.0, max_value=315.0,
            value=float(st.session_state.p_proc), step=0.1,
            help="Active machining process temperature (normal operating range: 305.7 - 313.8 K)"
        )
        machine_type = st.selectbox(
            "Product Variant Type",
            ["L", "M", "H"],
            index=["L", "M", "H"].index(st.session_state.p_type),
            help="Quality variant: Low (50%), Medium (30%), High (20%)"
        )

    with col_input2:
        st.markdown("##### ⚙️ Mechanical & Tool Telemetry")
        rpm = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1000, max_value=3000,
            value=int(st.session_state.p_rpm), step=10,
            help="Spindle rotational speed (normal range: 1168 - 2886 rpm)"
        )
        torque = st.number_input(
            "Torque [Nm]",
            min_value=3.0, max_value=80.0,
            value=float(st.session_state.p_torque), step=0.5,
            help="Torque delivered by the spindle motor (normal range: 3.8 - 76.6 Nm)"
        )
        wear = st.number_input(
            "Tool Wear Duration [min]",
            min_value=0, max_value=300,
            value=int(st.session_state.p_wear), step=1,
            help="Cumulative cutting tool usage in minutes (failure threshold typically > 200 min)"
        )

    # Derived physics indicators
    delta_t, power_w, overstrain = calculate_physics_indicators(air, process, rpm, torque, wear)

    with col_input3:
        st.markdown("##### 🔬 Real-Time Physics Telemetry")
        
        # Delta T
        delta_color = "normal" if delta_t >= 8.6 else "inverse"
        st.metric(
            "Temp Differential (ΔT)",
            f"{delta_t:.1f} K",
            delta="Normal Dissipation" if delta_t >= 8.6 else "Dissipation Risk (<8.6K)",
            delta_color=delta_color
        )
        
        # Mechanical Power
        power_kw = power_w / 1000.0
        p_status = "Nominal (3 - 9 kW)" if 3.0 <= power_kw <= 9.0 else "Power Over/Underload"
        p_color = "normal" if 3.0 <= power_kw <= 9.0 else "inverse"
        st.metric(
            "Calculated Motor Power",
            f"{power_kw:.2f} kW",
            delta=p_status,
            delta_color=p_color
        )
        
        # Tool Wear Condition
        wear_status = "Optimal" if wear < 150 else ("High Wear" if wear < 200 else "Critical Replacement")
        w_color = "normal" if wear < 200 else "inverse"
        st.metric(
            "Tool Wear Status",
            f"{wear} min",
            delta=wear_status,
            delta_color=w_color
        )

    st.markdown("---")

    # Algorithm Selection & Execution
    col_algo, col_btn = st.columns([3, 1])
    with col_algo:
        choices = list(bundle["models"].keys()) + ["Isolation Forest", "⭐ All Supervised Models (Consensus)"]
        # Default to XGBoost if present
        default_idx = choices.index("XGBoost") if "XGBoost" in choices else 0
        selected_model = st.selectbox("Select Machine Learning Model for Evaluation:", choices, index=default_idx)

    with col_btn:
        st.write("")
        st.write("")
        run_analysis = st.button("🚀 Analyze Telemetry", type="primary", use_container_width=True)

    # Execute Prediction
    if run_analysis:
        row = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear
        }])

        st.markdown("### 📊 Diagnostic Results")

        if selected_model == "⭐ All Supervised Models (Consensus)":
            # Multi-model consensus comparison
            cols_res = st.columns(len(bundle["models"]))
            probabilities = {}
            predictions = {}

            for idx, (m_name, model_obj) in enumerate(bundle["models"].items()):
                pred = int(model_obj.predict(row)[0])
                prob = float(model_obj.predict_proba(row)[0][list(model_obj.classes_).index(1)])
                predictions[m_name] = pred
                probabilities[m_name] = prob

            # Consensus summary
            fail_votes = sum(predictions.values())
            total_votes = len(predictions)
            avg_prob = np.mean(list(probabilities.values()))

            c_gauge, c_summary = st.columns([1, 1.4])
            with c_gauge:
                st.plotly_chart(render_gauge_chart(avg_prob, "Consensus Ensemble Score"), use_container_width=True)

            with c_summary:
                if fail_votes >= (total_votes / 2.0):
                    st.markdown("""
                    <div class='diag-box diag-box-danger'>
                        <h4 style='color: #ef4444; margin: 0;'>⚠️ CONSENSUS: MACHINE FAILURE DETECTED</h4>
                        <p style='margin: 6px 0 0 0; color: #cbd5e1;'>
                            Majority of evaluation models classify current parameters as a <b>Failure State</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='diag-box diag-box-safe'>
                        <h4 style='color: #10b981; margin: 0;'>✅ CONSENSUS: NOMINAL OPERATION</h4>
                        <p style='margin: 6px 0 0 0; color: #cbd5e1;'>
                            Majority of models predict normal operating conditions without immediate failure risk.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"**Model Votes for Failure:** `{fail_votes} of {total_votes}` models")
                st.markdown(f"**Average Failure Probability:** `{avg_prob:.1%}`")

            # Chart of all model probabilities
            df_consensus = pd.DataFrame({
                "Model": list(probabilities.keys()),
                "Failure Risk": [p * 100 for p in probabilities.values()],
                "Prediction": ["Failure" if p == 1 else "Normal" for p in predictions.values()]
            }).sort_values("Failure Risk", ascending=False)

            fig_consensus = px.bar(
                df_consensus,
                x="Model",
                y="Failure Risk",
                color="Prediction",
                color_discrete_map={"Failure": "#ef4444", "Normal": "#10b981"},
                title="Cross-Model Failure Probability Comparison",
                text=df_consensus["Failure Risk"].apply(lambda v: f"{v:.1f}%")
            )
            fig_consensus.update_layout(
                yaxis_range=[0, 100],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "sans-serif"}
            )
            st.plotly_chart(fig_consensus, use_container_width=True)

        elif selected_model == "Isolation Forest":
            transformed = bundle["isolation_preprocessor"].transform(row)
            is_anomaly = bundle["isolation_model"].predict(transformed)[0] == -1
            anomaly_score = -float(bundle["isolation_model"].score_samples(transformed)[0])

            col_iso1, col_iso2 = st.columns([1, 1.5])
            with col_iso1:
                st.metric("Anomaly Score (Offset)", f"{anomaly_score:.3f}")
                if is_anomaly:
                    st.markdown("<span class='status-pill pill-red'>Outlier Detected</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='status-pill pill-green'>In-Distribution</span>", unsafe_allow_html=True)

            with col_iso2:
                if is_anomaly:
                    st.markdown("""
                    <div class='diag-box diag-box-danger'>
                        <h4 style='color: #ef4444; margin: 0;'>⚠️ UNUSUAL OPERATING PATTERN</h4>
                        <p style='margin: 6px 0 0 0; color: #cbd5e1;'>
                            Isolation Forest flagged these sensor combinations as an atypical statistical outlier.
                            Unsupervised models identify novel abnormal states without direct supervision.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='diag-box diag-box-safe'>
                        <h4 style='color: #10b981; margin: 0;'>✅ NORMAL OPERATING REGIME</h4>
                        <p style='margin: 6px 0 0 0; color: #cbd5e1;'>
                            Operating telemetry fits within normal historical distribution patterns.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            st.caption("ℹ️ Note: Isolation Forest is an unsupervised anomaly detector and does not output calibrated failure probabilities.")

        else:
            # Single Supervised Model
            model = bundle["models"][selected_model]
            prediction = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0][list(model.classes_).index(1)])

            col_g, col_details = st.columns([1, 1.4])
            with col_g:
                st.plotly_chart(render_gauge_chart(probability, f"{selected_model} Score"), use_container_width=True)

            with col_details:
                if prediction == 1:
                    st.markdown("""
                    <div class='diag-box diag-box-danger'>
                        <h4 style='color: #ef4444; margin: 0;'>🚨 PREDICTED: MACHINE FAILURE CLASS</h4>
                        <p style='margin: 6px 0 0 0; color: #cbd5e1;'>
                            High confidence failure signature detected. Immediate condition-based maintenance inspection recommended.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='diag-box diag-box-safe'>
                        <h4 style='color: #10b981; margin: 0;'>✅ PREDICTED: NO FAILURE CLASS</h4>
                        <p style='margin: 6px 0 0 0; color: #cbd5e1;'>
                            Machine operating safely within acceptable parameters. Continue routine scheduled inspections.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"**Classification Decision:** `{'Failure (1)' if prediction == 1 else 'Nominal (0)'}`")
                st.markdown(f"**Model Confidence Probability:** `{probability:.2%}`")

        # Domain Physics Diagnostic Advisory
        st.markdown("#### 🛠️ Physics-Informed Diagnostic Advisory")
        advisory_items = []

        if wear >= 200:
            advisory_items.append((
                "🔴 Tool Wear Critical",
                f"Tool wear is at {wear} min (threshold: 200 min). Cutting tool requires immediate replacement or redressing to avoid catastrophic tool fracture (TWF).",
                "danger"
            ))
        elif wear >= 150:
            advisory_items.append((
                "🟠 Tool Wear Caution",
                f"Tool wear is at {wear} min. Schedule tool changeover during the next operational shift pause.",
                "warning"
            ))

        if delta_t < 8.6:
            advisory_items.append((
                "🔴 Heat Dissipation Failure (HDF) Risk",
                f"Process-Air temperature delta ΔT is {delta_t:.1f} K (< 8.6 K critical threshold). Check coolant delivery, flow rates, and heat exchanger fins.",
                "danger"
            ))

        if power_kw < 3.0 or power_kw > 9.0:
            advisory_items.append((
                "🟠 Power Overload / Underload (PWF) Risk",
                f"Spindle mechanical power is {power_kw:.2f} kW (standard envelope: 3.0 to 9.0 kW). Verify motor drives, voltage feed, and mechanical binding.",
                "warning"
            ))

        if overstrain > 11000:
            advisory_items.append((
                "🔴 Mechanical Overstrain (OSF) Risk",
                f"Torque × Tool Wear product is {overstrain:.0f} (dangerously elevated). High cutting resistance under worn tool state risks drive shaft deformation.",
                "danger"
            ))

        if not advisory_items:
            advisory_items.append((
                "🟢 All Physical Diagnostics Nominal",
                "Thermal dissipation, mechanical power envelope, and tool cutting stress are all within baseline limits.",
                "safe"
            ))

        for title_adv, desc_adv, level_adv in advisory_items:
            box_class = f"diag-box-{level_adv}"
            st.markdown(f"""
            <div class='diag-box {box_class}'>
                <strong>{title_adv}:</strong> {desc_adv}
            </div>
            """, unsafe_allow_html=True)

        st.caption("Educational demonstration prototype. Predictions must be verified by certified industrial maintenance personnel.")


# =============================================================================
# PAGE 2: MODEL BENCHMARKS & REPORTS
# =============================================================================
elif page == "📊 Model Benchmarks & Reports":
    st.title("📊 Model Benchmarks & Comprehensive Evaluation")
    st.markdown("""
    <div class='hero-banner'>
        <strong>Benchmarking Overview:</strong> Eight distinct algorithms were trained and evaluated on an independent 
        held-out test split (2,000 machines, 20% stratified test set). Models are compared across Precision, Recall, F1, Accuracy, and PR-AUC.
    </div>
    """, unsafe_allow_html=True)

    if metrics_df is not None:
        # Champion Model Card
        best_recall = metrics_df.loc[metrics_df["Recall"].idxmax()]
        best_f1 = metrics_df.loc[metrics_df["F1"].idxmax()]
        best_acc = metrics_df.loc[metrics_df["Accuracy"].idxmax()]

        c_b1, c_b2, c_b3 = st.columns(3)
        with c_b1:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>🏆 Champion Recall (Safety First)</div>
                <div class='kpi-value'>{best_recall['Model']}</div>
                <div class='kpi-sub'>Recall: <b>{best_recall['Recall']:.1%}</b> | Catches the most failures</div>
            </div>
            """, unsafe_allow_html=True)

        with c_b2:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>⚖️ Best Balanced F1-Score</div>
                <div class='kpi-value'>{best_f1['Model']}</div>
                <div class='kpi-sub'>F1-Score: <b>{best_f1['F1']:.4f}</b> | Precision: {best_f1['Precision']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with c_b3:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>🎯 Highest Overall Accuracy</div>
                <div class='kpi-value'>{best_acc['Model']}</div>
                <div class='kpi-sub'>Accuracy: <b>{best_acc['Accuracy']:.2%}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Interactive Performance Comparison")
        
        # Metric Selector for Chart
        metric_choice = st.selectbox(
            "Select Metric to Rank Models:",
            ["Recall", "F1", "Precision", "ROC_AUC", "PR_AUC", "Accuracy"],
            index=0,
            help="In predictive maintenance, Recall is paramount to avoid missing catastrophic failures."
        )

        sorted_df = metrics_df.sort_values(metric_choice, ascending=True)
        fig_bar = px.bar(
            sorted_df,
            x=metric_choice,
            y="Model",
            orientation="h",
            title=f"Models Ranked by {metric_choice}",
            color=metric_choice,
            color_continuous_scale="Viridis",
            text=sorted_df[metric_choice].apply(lambda x: f"{x:.4f}")
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            xaxis_range=[0, max(1.0, sorted_df[metric_choice].max() * 1.1)]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Full Metrics Data Table
        st.markdown("### 📋 Complete Benchmark Leaderboard")
        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1": "{:.4f}",
                "ROC_AUC": "{:.4f}",
                "PR_AUC": "{:.4f}"
            }),
            use_container_width=True
        )

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                "📥 Download Metrics CSV",
                metrics_df.to_csv(index=False),
                "model_metrics.csv",
                "text/csv",
                use_container_width=True
            )

        st.markdown("---")

        # Confusion Matrix & Classification Report Viewer
        st.markdown("### 🔍 Model Artifacts: Confusion Matrices & Detailed Reports")
        model_names_available = [m for m in metrics_df["Model"].tolist() if m != "Isolation Forest"]
        selected_matrix_model = st.selectbox("Inspect Detailed Report for Model:", model_names_available)

        safe_name = selected_matrix_model.lower().replace(" ", "_")
        cm_path = REPORTS_DIR / f"confusion_matrix_{safe_name}.png"
        cr_path = REPORTS_DIR / f"classification_report_{safe_name}.txt"

        col_img, col_txt = st.columns([1.1, 1])
        with col_img:
            if cm_path.exists():
                st.image(str(cm_path), caption=f"{selected_matrix_model} - Confusion Matrix", use_container_width=True)
            else:
                st.info("Confusion matrix image not found.")

        with col_txt:
            if cr_path.exists():
                st.markdown(f"**Classification Report: `{selected_matrix_model}`**")
                report_content = cr_path.read_text(encoding="utf-8")
                st.code(report_content, language="text")
            else:
                st.info("Classification report text file not found.")

    else:
        st.warning("Model metrics file not found. Please run `python train_model.py` to generate evaluation reports.")


# =============================================================================
# PAGE 3: BATCH FLEET PREDICTION
# =============================================================================
elif page == "📁 Batch Fleet Prediction":
    st.title("📁 Batch Machine Fleet Assessment")
    st.markdown("""
    <div class='hero-banner'>
        <strong>Fleet Monitoring:</strong> Run simultaneous predictive inference across multiple machines.
        Upload a factory CSV telemetry log or load a pre-configured sample fleet batch from the AI4I dataset.
    </div>
    """, unsafe_allow_html=True)

    tab_sample, tab_upload = st.tabs(["🚀 Quick Sample Fleet (25 Machines)", "📤 Upload Custom Telemetry CSV"])

    batch_df = None

    with tab_sample:
        st.write("Evaluate a randomized sample fleet of 25 industrial machines across product lines L, M, and H:")
        if st.button("Load & Run Sample Fleet", type="primary"):
            batch_df = load_sample_dataset(str(DATA_PATH), 25)

    with tab_upload:
        uploaded_file = st.file_uploader("Upload CSV Telemetry File", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

    if batch_df is not None:
        required_cols = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns in dataset: {missing_cols}")
        else:
            model_for_batch = st.selectbox(
                "Select Classifier for Fleet Assessment:",
                list(bundle["models"].keys()),
                index=list(bundle["models"].keys()).index("XGBoost") if "XGBoost" in bundle["models"] else 0
            )

            clf = bundle["models"][model_for_batch]
            input_features = batch_df[required_cols].copy()

            # Predict
            preds = clf.predict(input_features)
            probs = clf.predict_proba(input_features)[:, list(clf.classes_).index(1)]

            results_df = batch_df.copy()
            results_df["Predicted_Failure"] = preds
            results_df["Failure_Risk_Score"] = np.round(probs, 4)
            results_df["Status"] = ["🚨 High Risk" if p == 1 else "✅ Nominal" for p in preds]

            # KPI Summary
            total_machines = len(results_df)
            flagged_failures = int((results_df["Predicted_Failure"] == 1).sum())
            nominal_machines = total_machines - flagged_failures

            c_k1, c_k2, c_k3 = st.columns(3)
            with c_k1:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-title'>Total Fleet Machines</div>
                    <div class='kpi-value'>{total_machines}</div>
                    <div class='kpi-sub'>Monitored telemetry rows</div>
                </div>
                """, unsafe_allow_html=True)
            with c_k2:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-title'>Critical Action Required</div>
                    <div class='kpi-value' style='color: #ef4444;'>{flagged_failures}</div>
                    <div class='kpi-sub'>Machines flagged for maintenance</div>
                </div>
                """, unsafe_allow_html=True)
            with c_k3:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-title'>Nominal Operations</div>
                    <div class='kpi-value' style='color: #10b981;'>{nominal_machines}</div>
                    <div class='kpi-sub'>Healthy running status</div>
                </div>
                """, unsafe_allow_html=True)

            # Interactive Results Table
            st.markdown("#### 📋 Fleet Telemetry & Risk Scores")
            display_cols = ["Status", "Failure_Risk_Score"] + required_cols
            st.dataframe(results_df[display_cols], use_container_width=True)

            # Download Annotated Results
            st.download_button(
                "📥 Download Annotated Fleet Report (CSV)",
                results_df.to_csv(index=False),
                "fleet_maintenance_report.csv",
                "text/csv"
            )


# =============================================================================
# PAGE 4: DATASET & LEAKAGE ANALYSIS
# =============================================================================
elif page == "🔍 Dataset & Leakage Analysis":
    st.title("🔍 Dataset Exploration & Target Leakage Prevention")
    st.markdown("""
    <div class='hero-banner'>
        <strong>Methodological Rigor:</strong> Complete overview of the AI4I 2020 predictive maintenance benchmark dataset, 
        severe class imbalance handling, and architectural steps taken to prevent catastrophic target leakage.
    </div>
    """, unsafe_allow_html=True)

    c_d1, c_d2, c_d3, c_d4 = st.columns(4)
    with c_d1:
        st.metric("Total Dataset Instances", "10,000")
    with c_d2:
        st.metric("Failure Instances", "339", "3.39% Base Rate")
    with c_d3:
        st.metric("Input Features", "6 Physical Signals")
    with c_d4:
        st.metric("Target Leakage Features", "7 Excluded Flags")

    st.markdown("---")

    col_leak, col_eda = st.columns([1.2, 1])

    with col_leak:
        st.markdown("### 🛡️ Preventing Target Leakage")
        st.markdown("""
        In real-world predictive maintenance, models must forecast failure **before** it happens, 
        using only ambient and operating measurements.

        The AI4I dataset includes post-failure diagnostic labels:
        - `TWF`: Tool Wear Failure
        - `HDF`: Heat Dissipation Failure
        - `PWF`: Power Failure
        - `OSF`: Overstrain Failure
        - `RNF`: Random Failure
        - `UDI` & `Product ID`: Arbitrary row indices

        > ⚠️ **Critical Pipeline Rule:** All failure-mode flags were **strictly excluded** from training inputs. 
        > Including them creates near-100% artificial accuracy because they are recording what failed after the fact. 
        > Our model relies purely on physical sensor telemetry.
        """)

        st.markdown("#### Input Features Used in Inference:")
        st.markdown("""
        1. **Type**: Product quality tier (`L` = 50%, `M` = 30%, `H` = 20%)
        2. **Air temperature [K]**: Ambient sensor reading
        3. **Process temperature [K]**: Thermal sensor reading
        4. **Rotational speed [rpm]**: Spindle rotation frequency
        5. **Torque [Nm]**: Motor torque load
        6. **Tool wear [min]**: Duration tool has been actively cutting
        """)

    with col_eda:
        st.markdown("### 📊 Class Imbalance Distribution")
        fail_img_path = REPORTS_DIR / "failure_distribution.png"
        if fail_img_path.exists():
            st.image(str(fail_img_path), caption="Class Imbalance: 9,661 Normal vs 339 Failures", use_container_width=True)
        else:
            st.info("Run `python explore_data.py` to generate the distribution chart.")


# =============================================================================
# PAGE 5: ARCHITECTURE & GUIDANCE
# =============================================================================
else:
    st.title("📘 System Architecture & Operational Guidance")
    st.markdown("""
    <div class='hero-banner'>
        <strong>Project Blueprint:</strong> Infosys Springboard AI-Based Industrial Maintenance and Process Optimization System.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏗️ Machine Learning Pipeline Architecture")
    st.markdown("""
    ```text
    ┌─────────────────────────┐
    │  AI4I 2020 Raw Dataset  │ (10,000 instances, synthetic industrial plant data)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Target Leakage Filter   │ (Excludes UDI, Product ID, TWF, HDF, PWF, OSF, RNF)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Preprocessing Pipeline  │
    │  • Numerical Imputation │ (Median imputation + optional StandardScaler)
    │  • Categorical Encoding │ (OneHotEncoder for Product Variant: L, M, H)
    └───────────┬─────────────┘
                │
                ▼
    ┌────────────────────────────────────────────────────────┐
    │              8-Algorithm Benchmark Suite               │
    │  1. XGBoost                 5. Extra Trees             │
    │  2. Random Forest           6. AdaBoost                │
    │  3. Decision Tree           7. Gradient Boosting       │
    │  4. Logistic Regression     8. Isolation Forest        │
    └───────────┬────────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Streamlit UI Dashboard  │ (Real-time single prediction, fleet batch, physics advisory)
    └─────────────────────────┘
    ```
    """)

    st.markdown("### ⚠️ Industrial Safety & Operational Disclaimer")
    st.warning("""
    **Educational Prototype Notice:**
    This system is built as an educational demonstration prototype using the simulated AI4I 2020 dataset. 
    Predictions and physics rules do not replace certified safety instrumentation, vibration sensors, 
    or qualified maintenance engineers. Prior to deploying any ML model into active industrial plants, 
    thorough validation on actual factory equipment and hardware interlocks is mandatory.
    """)
