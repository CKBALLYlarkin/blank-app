# streamlit_app.py
# Fixed-Term Hours Planner (Working Document) — Streamlit
# Improvements added:
# - "Apply changes" button for planning tables (prevents uncommitted edits / non-updating)
# - hide_index=True for tables (removes left index column)
# - default leaver WTE to 1.0 when blank (reduces accidental 0.0)
#
# Core spec:
# - Spreadsheet-style (no certainty mode; no status column on leavers)
# - Part-time allocation lines include Increased Enrolment Allocation
# - Leavers list with Role dropdown (Teacher/Principal/Deputy Principal), no pre-population
# - Temporary absences split:
#     (A) Job-sharers list: one row per job-sharing teacher; WTE = count/2
#     (B) Career break / Secondment list: input current hours/week; app converts to WTE (hours/HoursPerPost)
# - Offset shown ALWAYS reflects: current P/CID + deployment in - leavers + assumed new CIDs - jobshare - CB/sec
# - Offset distribution grid appears only when offset is positive
# - Reserved appointments window (computed, not input):
#     * Principal vacancy reserve (if Principal leaving)
#     * Deputy Principal vacancy reserve (count of DPs leaving)
#     * Permanent vacancy capacity (if Perm Allocation > projected_perm_base)
#     * CIDs assumed (non-declined) as reserved WTE
# - Save/Load JSON
# - Schema-safe data_editor tables (prevents column mismatch errors)

import json
import streamlit as st
import pandas as pd

# -------------------------
# Constants
# -------------------------
HOURS_PER_POST_DEFAULT = 22.0
OFFSET_COL_NAME = "Offset (target)"
TOL_HOURS = 0.05

DEFAULT_PART_TIME_LINES = [
    "Ordinary Enrolment (Part-time)",
    "Increased Enrolment Allocation",
    "Junior Cycle Reform (Part-time)",
    "SEN Part-Time Allocation",
    "English as an Additional Language (EAL)",
    "Programme Coordinator Allocation",
    "SEN (Ukraine)",
    "Special Class",
    "Substitution",
]

DEFAULT_TEACHERS = ["Teacher A", "Teacher B", "Teacher C"]

LEAVER_ROLES = ["Teacher", "Principal", "Deputy Principal"]
LEAVER_TYPES = ["Retirement", "Resignation"]

ABSENCE_TYPES = ["Career break", "Secondment"]

CID_STATUSES = ["Not submitted", "Submitted", "Approved", "Declined/Not applicable"]


# -------------------------
# Helpers
# -------------------------
def ensure_columns(df: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
    """Guarantee df has exactly these columns (in order). Missing cols are added."""
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()


def to_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def offset_relevant(offset_hours_target: float) -> bool:
    return offset_hours_target > 0.0001


def ensure_alloc_schema(categories: list[str], teachers: list[str], include_offset: bool):
    """Only rebuild allocation matrix when schema changes."""
    desired_cols = list(teachers) + ([OFFSET_COL_NAME] if include_offset else [])
    desired_index = list(categories)

    if "alloc_df" not in st.session_state or st.session_state.alloc_df is None:
        st.session_state.alloc_df = pd.DataFrame(0.0, index=desired_index, columns=desired_cols)
        return

    old = st.session_state.alloc_df
    if list(old.index) == desired_index and list(old.columns) == desired_cols:
        return

    new = pd.DataFrame(0.0, index=desired_index, columns=desired_cols)
    common_rows = [r for r in old.index if r in new.index]
    common_cols = [c for c in old.columns if c in new.columns]
    if common_rows and common_cols:
        new.loc[common_rows, common_cols] = old.loc[common_rows, common_cols]
    st.session_state.alloc_df = new


def auto_distribute_offset_proportional(remaining_before_offset: pd.Series, target_hours: float) -> pd.Series:
    rem = remaining_before_offset.copy().clip(lower=0.0)
    if float(rem.sum()) <= 0 or target_hours <= 0:
        return pd.Series(0.0, index=rem.index)

    alloc = rem / float(rem.sum()) * target_hours

    for _ in range(10):
        over = alloc > rem
        if not over.any():
            break
        alloc[over] = rem[over]
        leftover = target_hours - float(alloc.sum())
        if leftover <= 1e-9:
            break
        room = (rem - alloc).clip(lower=0.0)
        room_total = float(room.sum())
        if room_total <= 1e-9:
            break
        alloc += room / room_total * leftover

    return alloc


# -------------------------
# Save / Load
# -------------------------
def export_state() -> str:
    data = {
        "hours_per_post": float(st.session_state.hours_per_post),
        "perm_allocation_wte": float(st.session_state.perm_allocation_wte),
        "perm_teachers_wte": float(st.session_state.perm_teachers_wte),
        "deployment_inward_wte": float(st.session_state.deployment_inward_wte),
        "part_time_df": st.session_state.part_time_df.to_dict(orient="records"),
        "leavers_df": st.session_state.leavers_df.to_dict(orient="records"),
        "jobsharers_df": st.session_state.jobsharers_df.to_dict(orient="records"),
        "absences_df": st.session_state.absences_df.to_dict(orient="records"),
        "cid_df": st.session_state.cid_df.to_dict(orient="records"),
        "teachers_df": st.session_state.teachers_df.to_dict(orient="records"),
        "alloc": {
            "index": list(st.session_state.alloc_df.index),
            "columns": list(st.session_state.alloc_df.columns),
            "data": st.session_state.alloc_df.values.tolist(),
        },
    }
    return json.dumps(data, indent=2)


def import_state(text: str):
    data = json.loads(text)

    st.session_state.hours_per_post = float(data.get("hours_per_post", HOURS_PER_POST_DEFAULT))
    st.session_state.perm_allocation_wte = float(data.get("perm_allocation_wte", 0.0))
    st.session_state.perm_teachers_wte = float(data.get("perm_teachers_wte", 0.0))
    st.session_state.deployment_inward_wte = float(data.get("deployment_inward_wte", 0.0))

    st.session_state.part_time_df = pd.DataFrame(data.get("part_time_df", []))
    st.session_state.leavers_df = pd.DataFrame(data.get("leavers_df", []))
    st.session_state.jobsharers_df = pd.DataFrame(data.get("jobsharers_df", []))
    st.session_state.absences_df = pd.DataFrame(data.get("absences_df", []))
    st.session_state.cid_df = pd.DataFrame(data.get("cid_df", []))
    st.session_state.teachers_df = pd.DataFrame(data.get("teachers_df", []))

    alloc = data.get("alloc", {})
    st.session_state.alloc_df = pd.DataFrame(
        alloc.get("data", []),
        index=alloc.get("index", []),
        columns=alloc.get("columns", []),
    )


# -------------------------
# Init
# -------------------------
def init_state():
    if "hours_per_post" not in st.session_state:
        st.session_state.hours_per_post = HOURS_PER_POST_DEFAULT

    if "perm_allocation_wte" not in st.session_state:
        st.session_state.perm_allocation_wte = 0.0
    if "perm_teachers_wte" not in st.session_state:
        st.session_state.perm_teachers_wte = 0.0
    if "deployment_inward_wte" not in st.session_state:
        st.session_state.deployment_inward_wte = 0.0

    if "part_time_df" not in st.session_state:
        st.session_state.part_time_df = pd.DataFrame(
            {
                "Part-time allocation line": DEFAULT_PART_TIME_LINES,
                "WTE (Part-time)": [0.0] * len(DEFAULT_PART_TIME_LINES),
            }
        )

    if "leavers_df" not in st.session_state:
        st.session_state.leavers_df = pd.DataFrame(columns=["Name/Label (optional)", "Role", "Leaving type", "WTE"])

    if "jobsharers_df" not in st.session_state:
        st.session_state.jobsharers_df = pd.DataFrame(columns=["Name/Label (optional)"])

    if "absences_df" not in st.session_state:
        st.session_state.absences_df = pd.DataFrame(columns=["Name/Label (optional)", "Absence type", "Current hours/week"])

    if "cid_df" not in st.session_state:
        st.session_state.cid_df = pd.DataFrame(columns=["Teacher (optional)", "CID hours/week", "Status"])

    if "teachers_df" not in st.session_state:
        st.session_state.teachers_df = pd.DataFrame({"Teacher": DEFAULT_TEACHERS})

    if "alloc_df" not in st.session_state:
        st.session_state.alloc_df = pd.DataFrame(0.0, index=DEFAULT_PART_TIME_LINES, columns=DEFAULT_TEACHERS)


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="Fixed-Term Hours Planner", layout="wide")
init_state()

st.title("Fixed-Term Hours Planner (Working Document)")
st.caption("Spreadsheet-style: edit figures whenever they change; click Apply for the planning tables.")

with st.sidebar:
    st.header("Settings")
    st.number_input("Hours per post", key="hours_per_post", min_value=1.0, max_value=30.0, step=0.5)

    if st.button("Reset app state (if something looks wrong)"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.subheader("Save / Load")
    st.download_button(
        "Save working document",
        data=export_state().encode("utf-8"),
        file_name="fixed_term_plan.json",
        mime="application/json",
    )
    upl = st.file_uploader("Load working document", type=["json"])
    if upl is not None:
        import_state(upl.read().decode("utf-8"))
        st.success("Loaded working document. If anything looks odd, press Reset once.")

hours_per_post = float(st.session_state.hours_per_post)

# -------------------------
# Baseline schedule inputs + Part-time lines
# -------------------------
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.subheader("1) Schedule baseline (Department documents)")
    st.number_input("Total Permanent Allocation (WTE) — Page 1", key="perm_allocation_wte", step=0.01)
    st.number_input("Total Permanent Teachers (incl. CIDs) (WTE) — Page 2", key="perm_teachers_wte", step=0.01)

    st.markdown("**Part-time allocation lines (WTE) — Page 1 part-time column**")
    PT_COLS = ["Part-time allocation line", "WTE (Part-time)"]
    st.session_state.part_time_df = ensure_columns(st.session_state.part_time_df, PT_COLS)

    pt_df = st.data_editor(
        st.session_state.part_time_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Part-time allocation line": st.column_config.TextColumn(required=True),
            "WTE (Part-time)": st.column_config.NumberColumn(min_value=0.0, step=0.01),
        },
        key="pt_editor",
    )
    st.session_state.part_time_df = pt_df

    pt_work = pt_df.copy()
    pt_work["Part-time allocation line"] = pt_work["Part-time allocation line"].fillna("").astype(str).str.strip()
    pt_work = pt_work[pt_work["Part-time allocation line"] != ""].copy()
    pt_work["WTE (Part-time)"] = to_num_series(pt_work["WTE (Part-time)"])

    total_part_time_wte = float(pt_work["WTE (Part-time)"].sum()) if len(pt_work) else 0.0
    total_part_time_hours = total_part_time_wte * hours_per_post
    st.info(f"**Total Part-time Allocation:** {total_part_time_wte:.2f} WTE  \n**= {total_part_time_hours:.2f} hours/week**")

with right:
    st.subheader("2) Planning inputs")
    st.number_input("Deployment inward (WTE) (optional)", key="deployment_inward_wte", step=0.01)

    st.markdown("---")
    st.write("Edit the tables below, then click **Apply changes** (this prevents lost edits).")

    # ---- Planning tables in a form (reliable commit)
    with st.form("planning_form", clear_on_submit=False):

        st.subheader("3) Leavers list (Retirements / Resignations)")
        LEAVERS_COLS = ["Name/Label (optional)", "Role", "Leaving type", "WTE"]
        leavers_base = ensure_columns(st.session_state.leavers_df, LEAVERS_COLS)

        # Default WTE to 1.0 where blank/invalid (helps non-coders)
        if len(leavers_base):
            leavers_base["WTE"] = pd.to_numeric(leavers_base["WTE"], errors="coerce").fillna(1.0)
        else:
            leavers_base["WTE"] = pd.Series(dtype=float)

        leavers_df = st.data_editor(
            leavers_base,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Name/Label (optional)": st.column_config.TextColumn(),
                "Role": st.column_config.SelectboxColumn(options=LEAVER_ROLES, required=True),
                "Leaving type": st.column_config.SelectboxColumn(options=LEAVER_TYPES, required=True),
                "WTE": st.column_config.NumberColumn(min_value=0.0, step=0.01),
            },
            key="leavers_editor",
        )

        st.markdown("---")
        st.subheader("4) Job-sharers list")
        st.caption("One row per job-sharing teacher. Count ÷ 2 = WTE.")
        JOB_COLS = ["Name/Label (optional)"]
        job_base = ensure_columns(st.session_state.jobsharers_df, JOB_COLS)

        jobsharers_df = st.data_editor(
            job_base,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={"Name/Label (optional)": st.column_config.TextColumn()},
            key="jobshare_editor",
        )

        st.markdown("---")
        st.subheader("5) Career break / Secondment list (hours/week)")
        st.caption("Enter current hours/week; converted to WTE using hours-per-post.")
        ABS_COLS = ["Name/Label (optional)", "Absence type", "Current hours/week"]
        abs_base = ensure_columns(st.session_state.absences_df, ABS_COLS)

        abs_df = st.data_editor(
            abs_base,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Name/Label (optional)": st.column_config.TextColumn(),
                "Absence type": st.column_config.SelectboxColumn(options=ABSENCE_TYPES, required=True),
                "Current hours/week": st.column_config.NumberColumn(min_value=0.0, step=0.5),
            },
            key="absences_editor",
        )

        st.markdown("---")
        st.subheader("6) CID pipeline (hours/week)")
        st.caption("For planning, all rows except Declined/Not applicable are assumed to be awarded.")
        CID_COLS = ["Teacher (optional)", "CID hours/week", "Status"]
        cid_base = ensure_columns(st.session_state.cid_df, CID_COLS)

        cid_df = st.data_editor(
            cid_base,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Teacher (optional)": st.column_config.TextColumn(),
                "CID hours/week": st.column_config.NumberColumn(min_value=0.0, step=0.5),
                "Status": st.column_config.SelectboxColumn(options=CID_STATUSES, required=True),
            },
            key="cid_editor",
        )

        applied = st.form_submit_button("Apply changes")

    if applied:
        st.session_state.leavers_df = leavers_df.reset_index(drop=True)
        st.session_state.jobsharers_df = jobsharers_df.reset_index(drop=True)
        st.session_state.absences_df = abs_df.reset_index(drop=True)
        st.session_state.cid_df = cid_df.reset_index(drop=True)
        st.success("Applied. Results updated.")
        st.rerun()

# -------------------------
# Compute totals
# -------------------------
perm_allocation = float(st.session_state.perm_allocation_wte)
current_pcid = float(st.session_state.perm_teachers_wte)
deployment_in = float(st.session_state.deployment_inward_wte)

# Leavers totals
leavers_work = st.session_state.leavers_df.copy()
if len(leavers_work):
    leavers_work["WTE"] = to_num_series(leavers_work["WTE"])
else:
    leavers_work["WTE"] = pd.Series(dtype=float)

retire_wte = float(leavers_work.loc[leavers_work["Leaving type"].eq("Retirement"), "WTE"].sum()) if len(leavers_work) else 0.0
resign_wte = float(leavers_work.loc[leavers_work["Leaving type"].eq("Resignation"), "WTE"].sum()) if len(leavers_work) else 0.0

principal_leaving_count = int((leavers_work["Role"].eq("Principal")).sum()) if len(leavers_work) else 0
dp_leaving_count = int((leavers_work["Role"].eq("Deputy Principal")).sum()) if len(leavers_work) else 0

# Job-share totals
jobshare_count = int(len(st.session_state.jobsharers_df)) if st.session_state.jobsharers_df is not None else 0
jobshare_wte = jobshare_count / 2.0

# Career break / secondment totals
abs_work = st.session_state.absences_df.copy()
if len(abs_work):
    abs_work["Current hours/week"] = to_num_series(abs_work["Current hours/week"])
else:
    abs_work["Current hours/week"] = pd.Series(dtype=float)

career_break_hours = float(abs_work.loc[abs_work["Absence type"].eq("Career break"), "Current hours/week"].sum()) if len(abs_work) else 0.0
secondment_hours = float(abs_work.loc[abs_work["Absence type"].eq("Secondment"), "Current hours/week"].sum()) if len(abs_work) else 0.0

career_break_wte = career_break_hours / hours_per_post if hours_per_post else 0.0
secondment_wte = secondment_hours / hours_per_post if hours_per_post else 0.0

# CIDs assumed (all non-declined)
cid_work = st.session_state.cid_df.copy()
if len(cid_work):
    cid_work["CID hours/week"] = to_num_series(cid_work["CID hours/week"])
else:
    cid_work["CID hours/week"] = pd.Series(dtype=float)

cid_hours_assumed = float(
    cid_work.loc[~cid_work["Status"].eq("Declined/Not applicable"), "CID hours/week"].sum()
) if len(cid_work) else 0.0
cid_wte_assumed = cid_hours_assumed / hours_per_post if hours_per_post else 0.0

# Effective P/CID (used for offset shown)
effective_pcid_next = (
    current_pcid
    + deployment_in
    - retire_wte
    - resign_wte
    + cid_wte_assumed
    - jobshare_wte
    - career_break_wte
    - secondment_wte
)

offset_wte_raw = effective_pcid_next - perm_allocation
effective_offset_wte = max(0.0, offset_wte_raw)
offset_hours_target = effective_offset_wte * hours_per_post
offset_needed = offset_relevant(offset_hours_target)

# Permanent vacancy capacity: perm allocation > projected perm base (no temporary absences)
projected_perm_base = current_pcid + deployment_in - retire_wte - resign_wte + cid_wte_assumed
perm_vacancy_wte = max(0.0, perm_allocation - projected_perm_base)

# Part-time hours
total_part_time_wte = float(pt_work["WTE (Part-time)"].sum()) if len(pt_work) else 0.0
total_part_time_hours = total_part_time_wte * hours_per_post

gross_hours_available = total_part_time_hours - offset_hours_target
net_hours_available = gross_hours_available - cid_hours_assumed  # reserve CID hours

# -------------------------
# Results
# -------------------------
st.divider()
st.subheader("7) Results (live)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Effective P/CID next year (WTE)", f"{effective_pcid_next:.2f}")
c2.metric("Effective offset (hours)", f"{offset_hours_target:.2f}")
c3.metric("Gross available (hours)", f"{gross_hours_available:.2f}")
c4.metric("Net available after CID reserve (hours)", f"{net_hours_available:.2f}")

with st.expander("Show calculation details", expanded=False):
    st.write(f"- Leavers: retire {retire_wte:.2f} WTE, resign {resign_wte:.2f} WTE")
    st.write(f"- Job-sharers: {jobshare_count} teachers → {jobshare_wte:.2f} WTE")
    st.write(f"- Career break: {career_break_hours:.1f} h/wk → {career_break_wte:.2f} WTE")
    st.write(f"- Secondment: {secondment_hours:.1f} h/wk → {secondment_wte:.2f} WTE")
    st.write(f"- New CIDs assumed: {cid_hours_assumed:.1f} h/wk → {cid_wte_assumed:.2f} WTE")
    st.write(
        f"- Offset WTE = (Effective P/CID {effective_pcid_next:.2f}) − (Perm allocation {perm_allocation:.2f}) "
        f"= {offset_wte_raw:.2f} → effective {effective_offset_wte:.2f}"
    )

# -------------------------
# Reserved appointments
# -------------------------
st.subheader("8) Reserved appointments (computed)")

reserved_rows: list[dict] = []
if principal_leaving_count > 0:
    reserved_rows.append({"Reserved appointment": "Principal vacancy", "WTE": float(principal_leaving_count)})
if dp_leaving_count > 0:
    reserved_rows.append({"Reserved appointment": "Deputy Principal vacancy", "WTE": float(dp_leaving_count)})
if perm_vacancy_wte > 0:
    reserved_rows.append({"Reserved appointment": "Permanent vacancy capacity", "WTE": float(perm_vacancy_wte)})
if cid_wte_assumed > 0:
    reserved_rows.append({"Reserved appointment": "CIDs (assumed to be awarded)", "WTE": float(cid_wte_assumed)})

if reserved_rows:
    st.dataframe(pd.DataFrame(reserved_rows), use_container_width=True, hide_index=True)
    if perm_vacancy_wte >= 1.0:
        st.warning(
            f"Permanent vacancy capacity is **{perm_vacancy_wte:.2f} WTE** (≥ 1.0). "
            "This indicates permanent appointments may be required."
        )
else:
    st.info("No reserved appointments identified from current inputs.")

# -------------------------
# Detailed allocation grid
# -------------------------
st.divider()
st.subheader("9) Detailed allocation grid (teachers × sources) + Offset distribution")

tabs = st.tabs(["A) Teacher list", "B) Allocation grid", "C) Checks & summaries"])

categories = pt_work["Part-time allocation line"].tolist() if len(pt_work) else []
cat_wte_map = dict(zip(pt_work["Part-time allocation line"], pt_work["WTE (Part-time)"])) if len(pt_work) else {}
cat_hours = pd.Series({k: float(cat_wte_map.get(k, 0.0)) * hours_per_post for k in categories})

with tabs[0]:
    st.write("Edit teacher names. Allocation grid rows come from your **Part-time allocation lines**.")
    TEACH_COLS = ["Teacher"]
    st.session_state.teachers_df = ensure_columns(st.session_state.teachers_df, TEACH_COLS)

    tdf = st.data_editor(
        st.session_state.teachers_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={"Teacher": st.column_config.TextColumn(required=True)},
        key="teachers_editor",
    )
    st.session_state.teachers_df = tdf

    teachers = (
        tdf["Teacher"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().tolist()
        if len(tdf) else []
    )

    if categories and teachers:
        ensure_alloc_schema(categories, teachers, include_offset=offset_needed)
        st.info(f"Teachers: {len(teachers)} {'(+ Offset column)' if offset_needed else ''}")
    elif not categories:
        st.warning("Add part-time allocation lines in section 1 to enable the grid.")
    else:
        st.warning("Add at least one teacher to enable the grid.")

with tabs[1]:
    teachers = (
        st.session_state.teachers_df["Teacher"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().tolist()
        if len(st.session_state.teachers_df) else []
    )

    if not categories:
        st.warning("No part-time allocation lines found.")
    elif not teachers:
        st.warning("No teachers found.")
    else:
        ensure_alloc_schema(categories, teachers, include_offset=offset_needed)

        if offset_needed:
            st.markdown("#### Offset distribution (required because offset is positive)")
            b1, b2, b3 = st.columns([1.0, 1.0, 2.0])
            with b1:
                do_auto = st.button("Auto-fill Offset (proportional)")
            with b2:
                clear_off = st.button("Clear Offset")
            with b3:
                st.write(f"Offset target: **{offset_hours_target:.2f} hours**")

            if do_auto or clear_off:
                alloc = st.session_state.alloc_df.copy()
                for c in alloc.columns:
                    alloc[c] = to_num_series(alloc[c])

                teacher_cols = [c for c in alloc.columns if c != OFFSET_COL_NAME]
                teachers_given = alloc[teacher_cols].sum(axis=1).reindex(categories).fillna(0.0)
                remaining_before_off = (cat_hours - teachers_given).reindex(categories).fillna(0.0)

                if clear_off and OFFSET_COL_NAME in st.session_state.alloc_df.columns:
                    st.session_state.alloc_df.loc[categories, OFFSET_COL_NAME] = 0.0

                if do_auto and OFFSET_COL_NAME in st.session_state.alloc_df.columns:
                    auto = auto_distribute_offset_proportional(remaining_before_off, offset_hours_target)
                    st.session_state.alloc_df.loc[categories, OFFSET_COL_NAME] = auto.values
        else:
            st.info("Offset is not positive, so Offset distribution is not required (Offset column hidden).")

        st.caption("Enter **hours/week** per teacher per source (supports contract wording).")
        alloc_df = st.data_editor(
            st.session_state.alloc_df,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(min_value=0.0, step=0.5)
                for c in st.session_state.alloc_df.columns
            },
            key="alloc_editor",
        )
        st.session_state.alloc_df = alloc_df

with tabs[2]:
    teachers = (
        st.session_state.teachers_df["Teacher"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().tolist()
        if len(st.session_state.teachers_df) else []
    )

    if not categories or not teachers:
        st.info("Add sources and teachers to see summaries.")
    else:
        alloc = st.session_state.alloc_df.copy()
        for c in alloc.columns:
            alloc[c] = to_num_series(alloc[c])

        teacher_cols = [c for c in alloc.columns if c != OFFSET_COL_NAME]

        teachers_given_by_cat = alloc[teacher_cols].sum(axis=1).reindex(categories).fillna(0.0)
        offset_by_cat = (
            alloc[OFFSET_COL_NAME].reindex(categories).fillna(0.0)
            if offset_needed and OFFSET_COL_NAME in alloc.columns
            else pd.Series(0.0, index=categories)
        )

        total_given_by_cat = teachers_given_by_cat + offset_by_cat
        remaining_by_cat = cat_hours - total_given_by_cat

        st.markdown("### Validation checks")

        if offset_needed:
            offset_alloc_sum = float(offset_by_cat.sum())
            diff = offset_alloc_sum - offset_hours_target
            if abs(diff) <= TOL_HOURS:
                st.success(f"Offset matches target (allocated {offset_alloc_sum:.2f}h vs target {offset_hours_target:.2f}h).")
            elif diff < 0:
                st.error(f"Offset UNDER-allocated by {abs(diff):.2f}h.")
            else:
                st.error(f"Offset OVER-allocated by {diff:.2f}h.")
        else:
            st.success("Offset not required (not positive).")

        over_alloc = remaining_by_cat < -TOL_HOURS
        if over_alloc.any():
            st.error("One or more sources are over-allocated:")
            for src, val in remaining_by_cat[over_alloc].items():
                st.write(f"- **{src}** is over by **{abs(float(val)):.2f} hours**")
        else:
            st.success("No source is over-allocated.")

        st.markdown("### Per-source summary")
        per_source_df = pd.DataFrame(
            {
                "Source": categories,
                "Available (h)": [float(cat_hours.get(k, 0.0)) for k in categories],
                "Teachers (h)": [float(teachers_given_by_cat.get(k, 0.0)) for k in categories],
                "Offset (h)": [float(offset_by_cat.get(k, 0.0)) for k in categories],
                "Remaining (h)": [float(remaining_by_cat.get(k, 0.0)) for k in categories],
            }
        )
        st.dataframe(per_source_df, use_container_width=True, hide_index=True)
