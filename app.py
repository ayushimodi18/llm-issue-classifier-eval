import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="LLM Issue Classifier Evaluation", page_icon="📊", layout="wide")

BASE = Path("Results")

st.title("📊 GitHub Issue Classification — LLM Evaluation Dashboard")
st.markdown("Compare language models for GitHub issue classification via an OpenAI-compatible inference API.")


def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def safe_load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


comparison_df = safe_load_csv(BASE / "model_comparison_parallel.csv")
if comparison_df is None:
    st.error("Missing Results/model_comparison_parallel.csv — run: python build_model_results.py parallel")
    st.stop()

models = comparison_df["model"].dropna().tolist()

# Column map — keyed by model name string
MODEL_COLUMN_MAP = {
    "gpt-4o-mini": {
        "pred": "gpt4o_prediction",
        "raw":  "gpt4o_raw_output",
        "json": BASE / "gpt4o_results_parallel.json",
    },
    "gpt-4.1-mini": {
        "pred": "gpt41_prediction",
        "raw":  "gpt41_raw_output",
        "json": BASE / "gpt41_results_parallel.json",
    },
    "gpt-5.1": {
        "pred": "gpt51_prediction",
        "raw":  "gpt51_raw_output",
        "json": BASE / "gpt51_results_parallel.json",
    },
}

# --------------------------------------------------
# Model selectors
# --------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    model_a = st.selectbox("Compare Model A", models, index=0)
with col2:
    model_b = st.selectbox("Compare Model B", models, index=min(1, len(models) - 1))

selected_a = comparison_df[comparison_df["model"] == model_a].iloc[0]
selected_b = comparison_df[comparison_df["model"] == model_b].iloc[0]

model_a_pred_col = MODEL_COLUMN_MAP.get(model_a, {}).get("pred")
model_a_raw_col  = MODEL_COLUMN_MAP.get(model_a, {}).get("raw")
model_b_pred_col = MODEL_COLUMN_MAP.get(model_b, {}).get("pred")
model_b_raw_col  = MODEL_COLUMN_MAP.get(model_b, {}).get("raw")

# --------------------------------------------------
# Overview
# --------------------------------------------------

st.divider()
st.header("Evaluation Overview")

ov1, ov2, ov3 = st.columns(3)
with ov1: st.metric("Total Models Evaluated", len(comparison_df))
with ov2: st.metric("Model A", model_a)
with ov3: st.metric("Model B", model_b)

# --------------------------------------------------
# All Candidate Results
# --------------------------------------------------

st.divider()
st.header("All Candidate Model Results")

candidate_cols = [c for c in [
    "model", "accuracy", "precision", "recall", "f1",
    "cost_per_call", "total_cost", "p50_latency", "p95_latency",
    "wall_clock_time", "throughput", "concurrency",
    "error_rate", "rate_limit_errors", "timeout_errors", "other_errors", "total_tokens"
] if c in comparison_df.columns]

st.dataframe(
    comparison_df[candidate_cols].sort_values(by="f1" if "f1" in comparison_df.columns else "model", ascending=False),
    use_container_width=True,
)

# --------------------------------------------------
# Operational Metrics
# --------------------------------------------------

st.divider()
st.header("Operational Metrics")

def render_ops(panel, label, row):
    with panel:
        st.markdown(f"### {label}")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Cost / Call",  f"${row['cost_per_call']:.8f}"   if "cost_per_call"  in row else "N/A")
        with c2: st.metric("Total Cost",   f"${row['total_cost']:.6f}"      if "total_cost"     in row else "N/A")
        with c3: st.metric("p50 Latency",  f"{row['p50_latency']:.2f}s"    if "p50_latency"   in row else "N/A")
        with c4: st.metric("p95 Latency",  f"{row['p95_latency']:.2f}s"    if "p95_latency"   in row else "N/A")
        c5, c6, c7, c8 = st.columns(4)
        with c5: st.metric("Concurrency",  f"{int(row['concurrency'])}"     if "concurrency"   in row and pd.notna(row["concurrency"]) else "N/A")
        with c6: st.metric("Wall Clock",   f"{row['wall_clock_time']:.2f}s" if "wall_clock_time" in row else "N/A")
        with c7: st.metric("Throughput",   f"{row['throughput']:.2f} req/s" if "throughput"    in row else "N/A")
        with c8: st.metric("Error Rate",   f"{row['error_rate']:.2f}%"      if "error_rate"    in row else "N/A")

render_ops(st.container(), model_a, selected_a)
st.markdown("---")
render_ops(st.container(), model_b, selected_b)

st.subheader("Error Breakdown")
st.dataframe(pd.DataFrame([
    {"model": model_a, "rate_limit": selected_a.get("rate_limit_errors", 0), "timeout": selected_a.get("timeout_errors", 0), "other": selected_a.get("other_errors", 0)},
    {"model": model_b, "rate_limit": selected_b.get("rate_limit_errors", 0), "timeout": selected_b.get("timeout_errors", 0), "other": selected_b.get("other_errors", 0)},
]), use_container_width=True)

st.subheader("Cost Trace")
st.caption("call_cost = (prompt_tokens/1k × input_rate) + (completion_tokens/1k × output_rate) — stored per row in prediction CSVs")
st.dataframe(pd.DataFrame([
    {"model": model_a, "total_tokens": selected_a.get("total_tokens"), "cost_per_call": selected_a.get("cost_per_call"), "total_cost": selected_a.get("total_cost")},
    {"model": model_b, "total_tokens": selected_b.get("total_tokens"), "cost_per_call": selected_b.get("cost_per_call"), "total_cost": selected_b.get("total_cost")},
]), use_container_width=True)

# --------------------------------------------------
# Scored / Unscored Tabs
# --------------------------------------------------

scored_tab, unscored_tab = st.tabs(["📊 Scored Corpus", "🤖 Unscored Corpus"])

# ---- SCORED TAB ----
with scored_tab:
    st.header("Scored Corpus Evaluation")
    st.info("Ground truth available. Metrics calculated against labeled issues.")

    metric_cols = [c for c in ["model", "accuracy", "precision", "recall", "f1"] if c in comparison_df.columns]
    st.subheader("Full Model Performance")
    st.dataframe(comparison_df[metric_cols], use_container_width=True)

    st.subheader("Selected Model Comparison")
    st.dataframe(pd.DataFrame([
        {"model": model_a, "accuracy": selected_a.get("accuracy"), "precision": selected_a.get("precision"), "recall": selected_a.get("recall"), "f1": selected_a.get("f1")},
        {"model": model_b, "accuracy": selected_b.get("accuracy"), "precision": selected_b.get("precision"), "recall": selected_b.get("recall"), "f1": selected_b.get("f1")},
    ]), use_container_width=True)

    # Per-class metrics
    st.subheader("Per-Class Precision / Recall / F1")

    json_a = safe_load_json(MODEL_COLUMN_MAP.get(model_a, {}).get("json", ""))
    json_b = safe_load_json(MODEL_COLUMN_MAP.get(model_b, {}).get("json", ""))

    def render_per_class(data, label):
        if not data or "per_class_metrics" not in data:
            st.info(f"No per-class data for {label}")
            return
        skip = {"accuracy", "macro avg", "weighted avg"}
        rows = [
            {"class": cls, "precision": round(v["precision"], 4), "recall": round(v["recall"], 4), "f1-score": round(v["f1-score"], 4), "support": int(v["support"])}
            for cls, v in data["per_class_metrics"].items() if cls not in skip
        ]
        st.markdown(f"**{label}**")
        st.dataframe(pd.DataFrame(rows).sort_values("class"), use_container_width=True)

    pc1, pc2 = st.columns(2)
    with pc1: render_per_class(json_a, model_a)
    with pc2: render_per_class(json_b, model_b)

    # Confusion matrices
    st.subheader("Confusion Matrix Comparison")
    if json_a and json_b:
        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown(f"### {model_a}")
            fig_a = px.imshow(json_a["confusion_matrix"], x=json_a["labels"], y=json_a["labels"], text_auto=True, aspect="auto", labels={"x": "Predicted", "y": "Actual"})
            st.plotly_chart(fig_a, use_container_width=True, key="cm_a")
        with cm2:
            st.markdown(f"### {model_b}")
            fig_b = px.imshow(json_b["confusion_matrix"], x=json_b["labels"], y=json_b["labels"], text_auto=True, aspect="auto", labels={"x": "Predicted", "y": "Actual"})
            st.plotly_chart(fig_b, use_container_width=True, key="cm_b")
    else:
        st.warning("Run build_model_results.py parallel to generate JSON files.")

    # Scored Disagreements
    st.divider()
    st.subheader("Scored Disagreements")

    scored_disagreements = safe_load_csv(BASE / "disagreements.csv")
    full_predictions_df  = safe_load_csv(BASE / "full_predictions.csv")

    if scored_disagreements is not None and full_predictions_df is not None:
        req = [c for c in ["title", "ground_truth", model_a_pred_col, model_b_pred_col] if c]
        if all(c in scored_disagreements.columns for c in req):
            display_df = scored_disagreements[req].rename(columns={
                model_a_pred_col: f"{model_a} prediction",
                model_b_pred_col: f"{model_b} prediction",
            })
            search = st.text_input("Search Scored Issues", key="scored_search")
            if search:
                display_df = display_df[display_df["title"].str.contains(search, case=False, na=False)]
            st.dataframe(display_df, use_container_width=True)

            titles = scored_disagreements["title"].dropna().tolist()
            if titles:
                sel = st.selectbox("Drill down", titles, key="scored_sel")
                row = full_predictions_df[full_predictions_df["title"] == sel]
                if not row.empty:
                    r = row.iloc[0]
                    with st.expander("Prediction Details"):
                        st.success(f"Ground Truth: {r.get('ground_truth', 'N/A')}")
                        d1, d2 = st.columns(2)
                        with d1:
                            st.markdown(f"**{model_a}** → {r.get(model_a_pred_col, '')}")
                            st.text_area("Raw Output", value=r.get(model_a_raw_col, ""), height=180, disabled=True, key="sc_ra")
                        with d2:
                            st.markdown(f"**{model_b}** → {r.get(model_b_pred_col, '')}")
                            st.text_area("Raw Output", value=r.get(model_b_raw_col, ""), height=180, disabled=True, key="sc_rb")
        else:
            st.info("Disagreement file missing required columns for the selected model pair.")
    else:
        st.info("Run build_disagreements.py and build_full_predictions.py first.")


# ---- UNSCORED TAB ----
with unscored_tab:
    st.header("Unscored Corpus Evaluation")
    st.info("No ground truth. Showing model agreement and prediction analysis.")

    agreement_stats   = safe_load_json(BASE / "unscored_agreement_stats.json")
    distribution_data = safe_load_json(BASE / "unscored_class_distribution.json")
    unscored_preds    = safe_load_csv(BASE / "unscored_predictions.csv")
    unscored_disagree = safe_load_csv(BASE / "unscored_disagreements.csv")

    st.subheader("Agreement Summary")
    a1, a2, a3 = st.columns(3)
    with a1: st.metric("Agreement Rate",     f"{agreement_stats['agreement_rate']:.2f}%"  if agreement_stats and "agreement_rate"    in agreement_stats else "N/A")
    with a2: st.metric("Agreement Count",    agreement_stats["agreement_count"]            if agreement_stats and "agreement_count"   in agreement_stats else "N/A")
    with a3: st.metric("Disagreement Count", agreement_stats["disagreement_count"]         if agreement_stats and "disagreement_count" in agreement_stats else "N/A")

    st.divider()
    st.subheader("Class Distribution Comparison")

    DIST_MAP = {
        "gpt-4o-mini":  "gpt4o_prediction",
        "gpt-4.1-mini": "gpt41_prediction",
        "gpt-5.1":      "gpt51_prediction",
    }

    if distribution_data:
        dist_a = distribution_data.get(DIST_MAP.get(model_a, ""), {})
        dist_b = distribution_data.get(DIST_MAP.get(model_b, ""), {})
        if dist_a and dist_b:
            dc1, dc2 = st.columns(2)
            with dc1:
                st.plotly_chart(px.bar(pd.DataFrame({"label": list(dist_a.keys()), "count": list(dist_a.values())}), x="label", y="count", title=model_a), use_container_width=True, key="dist_a")
            with dc2:
                st.plotly_chart(px.bar(pd.DataFrame({"label": list(dist_b.keys()), "count": list(dist_b.values())}), x="label", y="count", title=model_b), use_container_width=True, key="dist_b")
        else:
            st.info("Run build_unscored_distribution.py to enable this chart.")

    st.divider()
    st.subheader("Unscored Disagreements")

    if unscored_disagree is not None and unscored_preds is not None:
        req = [c for c in ["title", model_a_pred_col, model_b_pred_col] if c]
        if all(c in unscored_disagree.columns for c in req):
            ud_df = unscored_disagree[req].rename(columns={
                model_a_pred_col: f"{model_a} prediction",
                model_b_pred_col: f"{model_b} prediction",
            })
            us_search = st.text_input("Search Unscored Issues", key="us_search")
            if us_search:
                ud_df = ud_df[ud_df["title"].str.contains(us_search, case=False, na=False)]
            st.dataframe(ud_df, use_container_width=True, height=350)

            ud_titles = unscored_disagree["title"].dropna().tolist()
            if ud_titles:
                us_sel = st.selectbox("Drill down", ud_titles, key="us_sel")
                us_row = unscored_preds[unscored_preds["title"] == us_sel]
                if not us_row.empty:
                    ur = us_row.iloc[0]
                    with st.expander("Unscored Prediction Details"):
                        u1, u2 = st.columns(2)
                        with u1:
                            st.markdown(f"**{model_a}** → {ur.get(model_a_pred_col, '')}")
                            st.text_area("Raw Output", value=ur.get(model_a_raw_col, ""), height=180, disabled=True, key="us_ra")
                        with u2:
                            st.markdown(f"**{model_b}** → {ur.get(model_b_pred_col, '')}")
                            st.text_area("Raw Output", value=ur.get(model_b_raw_col, ""), height=180, disabled=True, key="us_rb")
        else:
            st.info("Disagreement file missing required columns for selected model pair.")
    else:
        st.info("Run build_unscored_disagreements.py and build_unscored_predictions.py first.")

    st.divider()
    st.subheader("Full Unscored Predictions")
    if unscored_preds is not None:
        fp_cols = [c for c in ["title", model_a_pred_col, model_b_pred_col] if c and c in unscored_preds.columns]
        fp_df = unscored_preds[fp_cols].rename(columns={
            model_a_pred_col: f"{model_a} prediction",
            model_b_pred_col: f"{model_b} prediction",
        })
        fp_search = st.text_input("Search All Unscored", key="fp_search")
        if fp_search:
            fp_df = fp_df[fp_df["title"].str.contains(fp_search, case=False, na=False)]
        st.dataframe(fp_df, use_container_width=True, height=400)

    st.divider()
    st.subheader("Full Unscored Prediction Drill Down")
    if unscored_preds is not None and "title" in unscored_preds.columns:
        all_titles = unscored_preds["title"].dropna().tolist()
        if all_titles:
            sel_full = st.selectbox("Select Any Unscored Issue", all_titles, key="full_us_sel")
            full_row = unscored_preds[unscored_preds["title"] == sel_full]
            if not full_row.empty:
                fr = full_row.iloc[0]
                with st.expander("View Raw Outputs"):
                    fu1, fu2 = st.columns(2)
                    with fu1:
                        st.markdown(f"**{model_a}** → {fr.get(model_a_pred_col, '')}")
                        st.text_area("Raw Output", value=fr.get(model_a_raw_col, ""), height=220, disabled=True, key="full_us_ra")
                    with fu2:
                        st.markdown(f"**{model_b}** → {fr.get(model_b_pred_col, '')}")
                        st.text_area("Raw Output", value=fr.get(model_b_raw_col, ""), height=220, disabled=True, key="full_us_rb")


# --------------------------------------------------
# Production Recommendation — dynamic fallback
# --------------------------------------------------

st.divider()
st.header("🏆 Production Recommendation")

rec_df = comparison_df.copy()
if "f1" in rec_df.columns and "total_cost" in rec_df.columns:
    rec_df["cost_per_correct"] = rec_df["total_cost"] / rec_df["accuracy"].clip(lower=1e-9)
else:
    rec_df["cost_per_correct"] = None

rec_df["production_score"] = rec_df["f1"] if "f1" in rec_df.columns else 0
sorted_rec   = rec_df.sort_values(by="production_score", ascending=False)
best_model   = sorted_rec.iloc[0]
fallback_row = sorted_rec.iloc[1] if len(sorted_rec) > 1 else None

st.success(f"✅ Recommended Primary Model: **{best_model['model']}**")
if fallback_row is not None:
    st.info(f"🔁 Recommended Fallback Model: **{fallback_row['model']}**")

r1, r2, r3, r4 = st.columns(4)
with r1: st.metric("Accuracy",       f"{best_model['accuracy']:.4f}"          if "accuracy"         in best_model else "N/A")
with r2: st.metric("F1 Score",       f"{best_model['f1']:.4f}"                if "f1"               in best_model else "N/A")
with r3: st.metric("p95 Latency",    f"{best_model['p95_latency']:.2f}s"      if "p95_latency"      in best_model else "N/A")
with r4: st.metric("Cost/Correct",   f"${best_model['cost_per_correct']:.8f}" if pd.notna(best_model.get("cost_per_correct")) else "N/A")

st.subheader("Model Ranking")
rank_cols = [c for c in ["model", "accuracy", "f1", "cost_per_call", "total_cost", "cost_per_correct", "p95_latency", "throughput", "error_rate", "production_score"] if c in rec_df.columns]
st.dataframe(rec_df[rank_cols].sort_values(by="production_score", ascending=False), use_container_width=True)

with st.expander("Why this recommendation", expanded=True):
    fallback_name = fallback_row["model"] if fallback_row is not None else "N/A"
    st.markdown(f"""
**Primary: {best_model['model']}**

Selected on highest weighted F1 across the scored evaluation corpus. Full cost, latency, and throughput visible in the table above.

**Fallback: {fallback_name}**

Routes to fallback on: ambiguous/malformed outputs, schema violations, rate-limit exhaustion, and security-class issues.

**Suggested production pattern**
1. Default classifier → primary model.
2. Output not in valid label set → re-run on fallback.
3. Persistent failures or `security` label → human review queue.
4. Monitor: disagreement rate drift, class distribution shift, cost per correct classification, p95 latency regression.
""")

with st.expander("Production Failure Policy"):
    st.markdown("""
- Retry individual failed requests (not batches) — per-issue retry is the default pattern.
- Rate limit (429): exponential backoff → retry → fallback model.
- Timeout: single retry → human review.
- `security` class: always human review regardless of model confidence.
- Re-evaluate when disagreement rate shifts >5% or class distribution drifts materially.
- Cost alerting: set per-run budget cap; alert on cost_per_correct regression.
""")
