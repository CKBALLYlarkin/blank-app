# streamlit_app.py
# Fixed-Term Hours Planner (Working Document) — Streamlit
#
# Key logic (updated):
# - Effective P/CID next year (for offset/surplus) does NOT subtract job-share/CB/secondment.
# - Job-share/CB/secondment create RELEASED HOURS to spend (allocate to teachers and/or offset).
# - Released hours appear as extra "source" rows in the allocation grid:
#     Released hours - Job-share
#     Released hours - Career break
#     Released hours - Secondment
# - Allocation grid shows Available/Allocated/Remaining on the SAME screen.
# - Grid accepts time entry like 10:30 (hours:minutes) as well as decimals.
# - Planning tables use Apply button to avoid lost/uncommitted edits.

import json
import re
import streamlit as st
import pandas as pd

# -------------------------
# Constants
# -------------------------
HOURS_PER_POST_DEFAULT = 22.0
OFFSET_COL_NAME = "Offset (target)"
TOL_HOURS = 0.05

# Source row names for released hours
REL_JS = "Released hours - Job-share"
REL_CB = "Released hours - Career break"
REL_SEC = "Released hours - Secondment"

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


def parse_time_to_hours(x) -> float:
    """
    Accepts:
      - 10            -> 10.0
      - 10.5          -> 10.5
      - "10:30"       -> 10.5
      - "10h 30m"     -> 10.5
      - "90m"         -> 1.5
      - "" / None     -> 0.0
    Returns hours as float.
    """
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return 0.0

    s = str(x).strip().lower()
    if s == "":
        return 0.0

    # H:MM
    m = re.match(r"^(\d+)\s*:\s*(\d{1,2})$", s)
    if m:
        h = int(m.group(1))
        mins = int(m.group(2))
        mins = max(0, min(mins, 59))
        return h + mins / 60.0

    # "10h 30m" or "10h" or "30m"
    m = re.match(r"^(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?$", s)
    if m and (m.group(1) or m.group(2)):
        h = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        return h + mins / 60.0

    # "90m"
    m = re.match(r"^(\d+)\s*m$", s)
    if m:
        return int(m.group(1)) / 60.0

    # Fall back to float
    try:
        return float(s)
    except Exception:
        return 0.0


def df_to_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a dataframe of mixed inputs into numeric hours."""
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].apply(parse_time_to_hours)
    return out


def hours_to_hmm(hours: float) -> str:
    """Format float hours as H:MM."""
    try:
        total_mins = int(round(float(hours) * 60))
    except Exception:
        return "0:00"
    h = total_mins // 60
    m = total_mins % 60
    return f"{h}:{m:02d}"


def ensure_alloc_schema(categories: list[str], teachers: list[str], include_offset: bool):
    """Only rebuild allocation matrix when schema changes."""
    desired_cols = list(teachers) + ([OFFSET_COL_NAME] if include_offset else [])
    desired_index = list(categories)

    if "alloc_df" not in st.session_state or st.session_state.alloc_df is None:
        st.session_state.alloc_df = pd.DataFrame("", index=desired_index, columns=desired_cols)
        return

    old = st.session_state.alloc_df
    if list(old.index) == desired_index and list(old.columns) == desired_cols:
        return

    new = pd.DataFrame("", index=desired_index, columns=desired_cols)
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
            {"Part-time allocation line": DEFAULT_PART_TIME_LINES, "WTE (Part-time)": [0.0] * len(DEFAULT_PART_TIME_LINES)}
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
        st.session_state.alloc_df = pd.DataFrame("", index=DEFAULT_PART_TIME_LINES, columns=DEFAULT_TEACHERS)


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="Fixed-Term Hours Planner", layout="wide")
init_state()

st.title("Fixed-Term Hours Planner (Working Document)")
st.caption("Edit figures whenever they change. Use **Apply changes** for planning tables. Allocation grid accepts 10:30 format.")

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

left, right = st.columns([1.25, 1.0], gap="large")

# -------------------------
# Left: schedule + part-time lines
# -------------------------
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

# -------------------------
# Right: planning tables (apply button)
# -------------------------
with right:
    st.subheader("2) Planning inputs")
    st.number_input("Deployment inward (WTE) (optional)", key="deployment_inward_wte", step=0.01)

    st.markdown("---")
    st.write("Edit the tables below, then click **Apply changes** (prevents lost edits).")

    with st.form("planning_form", clear_on_submit=False):

        st.subheader("3) Leavers list (Retirements / Resignations)")
        LEAVERS_COLS = ["Name/Label (optional)", "Role", "Leaving type", "WTE"]
        leavers_base = ensure_columns(st.session_state.leavers_df, LEAVERS_COLS)

        # Default WTE to 1.0 if blank/invalid
        if len(leavers_base):
            leavers_base["WTE"] = pd.to_numeric(leavers_base["WTE"], errors="coerce").fillna(1.0)

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
        st.caption("One row per job-sharing teacher. Each job-share releases 0.5 post worth of hours.")
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
        st.caption("Enter the teacher’s current hours/week. These hours are released and must be allocated.")
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

# Job-share released hours
jobshare_count = int(len(st.session_state.jobsharers_df)) if st.session_state.jobsharers_df is not None else 0
jobshare_released_hours = jobshare_count * (hours_per_post / 2.0)

# Career break / secondment released hours
abs_work = st.session_state.absences_df.copy()
if len(abs_work):
    abs_work["Current hours/week"] = to_num_series(abs_work["Current hours/week"])
else:
    abs_work["Current hours/week"] = pd.Series(dtype=float)

career_break_hours = float(abs_work.loc[abs_work["Absence type"].eq("Career break"), "Current hours/week"].sum()) if len(abs_work) else 0.0
secondment_hours = float(abs_work.loc[abs_work["Absence type"].eq("Secondment"), "Current hours/week"].sum()) if len(abs_work) else 0.0

released_hours_total = jobshare_released_hours + career_break_hours + secondment_hours

# CIDs assumed
cid_work = st.session_state.cid_df.copy()
if len(cid_work):
    cid_work["CID hours/week"] = to_num_series(cid_work["CID hours/week"])
else:
    cid_work["CID hours/week"] = pd.Series(dtype=float)

cid_hours_assumed = float(
    cid_work.loc[~cid_work["Status"].eq("Declined/Not applicable"), "CID hours/week"].sum()
) if len(cid_work) else 0.0
cid_wte_assumed = cid_hours_assumed / hours_per_post if hours_per_post else 0.0

# Effective P/CID next year (DO NOT subtract job-share / CB / secondment)
effective_pcid_next = (
    current_pcid
    + deployment_in
    - retire_wte
    - resign_wte
    + cid_wte_assumed
)

# Offset
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

# Availability (released hours must be allocated too)
gross_hours_available = total_part_time_hours + released_hours_total - offset_hours_target
net_hours_available = gross_hours_available - cid_hours_assumed  # reserve CID hours

# -------------------------
# Results
# -------------------------
st.divider()
st.subheader("7) Results (live)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Effective P/CID next year (WTE)", f"{effective_pcid_next:.2f}")
c2.metric("Offset target (hours)", f"{offset_hours_target:.2f}")
c3.metric("Gross available incl. released (hours)", f"{gross_hours_available:.2f}")
c4.metric("Net available after CID reserve (hours)", f"{net_hours_available:.2f}")

with st.expander("Show calculation details", expanded=False):
    st.write(f"- Leavers: retire {retire_wte:.2f} WTE, resign {resign_wte:.2f} WTE")
    st.write(f"- Job-share count: {jobshare_count} → released {jobshare_released_hours:.2f} hours ({hours_to_hmm(jobshare_released_hours)})")
    st.write(f"- Career break released: {career_break_hours:.2f} hours ({hours_to_hmm(career_break_hours)})")
    st.write(f"- Secondment released: {secondment_hours:.2f} hours ({hours_to_hmm(secondment_hours)})")
    st.write(f"- New CIDs assumed: {cid_hours_assumed:.2f} hours → {cid_wte_assumed:.2f} WTE")
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

# Base categories from part-time schedule lines
pt_categories = pt_work["Part-time allocation line"].tolist() if len(pt_work) else []

# Add released rows (always present so principals can plan; they may be zero)
categories = pt_categories + [REL_JS, REL_CB, REL_SEC]

# Available hours per source:
cat_hours = pd.Series({k: 0.0 for k in categories})
# Part-time sources: WTE × hours_per_post
if len(pt_work):
    for _, r in pt_work.iterrows():
        cat_hours[str(r["Part-time allocation line"])] = float(r["WTE (Part-time)"]) * hours_per_post
# Released sources: direct hours
cat_hours[REL_JS] = float(jobshare_released_hours)
cat_hours[REL_CB] = float(career_break_hours)
cat_hours[REL_SEC] = float(secondment_hours)

with tabs[0]:
    st.write("Edit teacher names. Allocation grid rows come from your part-time lines PLUS released hours rows.")
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
        if len(tdf)
        else []
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
        if len(st.session_state.teachers_df)
        else []
    )

    if not categories:
        st.warning("No allocation lines found.")
    elif not teachers:
        st.warning("No teachers found.")
    else:
        ensure_alloc_schema(categories, teachers, include_offset=offset_needed)

        # Calculate hours from current grid values (supports 10:30 input)
        alloc_calc = df_to_hours(st.session_state.alloc_df)
        teacher_cols = [c for c in alloc_calc.columns if c != OFFSET_COL_NAME]

        teachers_given_by_cat = alloc_calc[teacher_cols].sum(axis=1).reindex(categories).fillna(0.0)
        offset_by_cat = (
            alloc_calc[OFFSET_COL_NAME].reindex(categories).fillna(0.0)
            if offset_needed and OFFSET_COL_NAME in alloc_calc.columns
            else pd.Series(0.0, index=categories)
        )

        available_by_cat = cat_hours.reindex(categories).fillna(0.0)
        allocated_by_cat = teachers_given_by_cat + offset_by_cat
        remaining_by_cat = available_by_cat - allocated_by_cat

        left_sum, right_grid = st.columns([0.45, 0.55], gap="large")

        with left_sum:
            st.markdown("#### Source hours (Available / Allocated / Remaining)")
            summary_df = pd.DataFrame(
                {
                    "Source": categories,
                    "Available (h)": [float(available_by_cat.get(k, 0.0)) for k in categories],
                    "Available (H:MM)": [hours_to_hmm(float(available_by_cat.get(k, 0.0))) for k in categories],
                    "Allocated (h)": [float(allocated_by_cat.get(k, 0.0)) for k in categories],
                    "Allocated (H:MM)": [hours_to_hmm(float(allocated_by_cat.get(k, 0.0))) for k in categories],
                    "Remaining (h)": [float(remaining_by_cat.get(k, 0.0)) for k in categories],
                    "Remaining (H:MM)": [hours_to_hmm(float(remaining_by_cat.get(k, 0.0))) for k in categories],
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            if offset_needed:
                st.markdown("#### Offset distribution controls")
                b1, b2 = st.columns(2)
                with b1:
                    do_auto = st.button("Auto-fill Offset (proportional)")
                with b2:
                    clear_off = st.button("Clear Offset")
                st.write(f"Offset target: **{offset_hours_target:.2f} hours** ({hours_to_hmm(offset_hours_target)})")

                if do_auto or clear_off:
                    alloc = df_to_hours(st.session_state.alloc_df)

                    teacher_cols2 = [c for c in alloc.columns if c != OFFSET_COL_NAME]
                    teachers_given2 = alloc[teacher_cols2].sum(axis=1).reindex(categories).fillna(0.0)
                    remaining_before_off = (available_by_cat - teachers_given2).reindex(categories).fillna(0.0)

                    if clear_off and OFFSET_COL_NAME in st.session_state.alloc_df.columns:
                        st.session_state.alloc_df.loc[categories, OFFSET_COL_NAME] = ""

                    if do_auto and OFFSET_COL_NAME in st.session_state.alloc_df.columns:
                        auto = auto_distribute_offset_proportional(remaining_before_off, offset_hours_target)
                        # Store as H:MM strings (nicer) — but decimals also fine
                        st.session_state.alloc_df.loc[categories, OFFSET_COL_NAME] = [hours_to_hmm(v) for v in auto.values]

                    st.rerun()
            else:
                st.info("Offset is not positive, so Offset distribution is not required.")

        with right_grid:
            st.caption("Enter hours as **H:MM** (e.g., 10:30) or decimals (e.g., 10.5).")
            alloc_df = st.data_editor(
                st.session_state.alloc_df,
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                column_config={
                    c: st.column_config.TextColumn(help="Examples: 10, 10:30, 10.5, 90m")
                    for c in st.session_state.alloc_df.columns
                },
                key="alloc_editor",
            )
            st.session_state.alloc_df = alloc_df

with tabs[2]:
    teachers = (
        st.session_state.teachers_df["Teacher"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().tolist()
        if len(st.session_state.teachers_df)
        else []
    )

    if not categories or not teachers:
        st.info("Add sources and teachers to see summaries.")
    else:
        alloc = df_to_hours(st.session_state.alloc_df)
        teacher_cols = [c for c in alloc.columns if c != OFFSET_COL_NAME]

        teachers_given_by_cat = alloc[teacher_cols].sum(axis=1).reindex(categories).fillna(0.0)
        offset_by_cat = (
            alloc[OFFSET_COL_NAME].reindex(categories).fillna(0.0)
            if offset_needed and OFFSET_COL_NAME in alloc.columns
            else pd.Series(0.0, index=categories)
        )

        available_by_cat = cat_hours.reindex(categories).fillna(0.0)
        total_given_by_cat = teachers_given_by_cat + offset_by_cat
        remaining_by_cat = available_by_cat - total_given_by_cat

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
                st.write(f"- **{src}** is over by **{abs(float(val)):.2f} hours** ({hours_to_hmm(abs(float(val)))})")
        else:
            st.success("No source is over-allocated.")

        st.markdown("### Per-teacher totals (hours/week)")
        teacher_totals = alloc[teacher_cols].sum(axis=0)
        per_teacher_df = pd.DataFrame(
            {
                "Teacher": teacher_totals.index,
                "Total (h)": [float(teacher_totals[t]) for t in teacher_totals.index],
                "Total (H:MM)": [hours_to_hmm(float(teacher_totals[t])) for t in teacher_totals.index],
            }
        ).sort_values("Teacher")
        st.dataframe(per_teacher_df, use_container_width=True, hide_index=True)

        st.markdown("### Contract-style breakdown (copy/paste)")
        for teacher in teacher_cols:
            series = alloc[teacher].reindex(categories).fillna(0.0)
            nz = series[series > 0.0].sort_values(ascending=False)
            total = float(series.sum())
            if total <= 0:
                continue
            parts = [f"{src} {hours_to_hmm(float(hrs))}" for src, hrs in nz.items()]
            st.code(f"{teacher}: {hours_to_hmm(total)} (" + ", ".join(parts) + ")", language="text")

        if offset_needed:
            st.markdown("### Offset breakdown (copy/paste)")
            nz = offset_by_cat[offset_by_cat > 0.0].sort_values(ascending=False)
            total = float(offset_by_cat.sum())
            parts = [f"{src} {hours_to_hmm(float(hrs))}" for src, hrs in nz.items()]
            st.code(f"Offset: {hours_to_hmm(total)} (" + ", ".join(parts) + ")", language="text")
