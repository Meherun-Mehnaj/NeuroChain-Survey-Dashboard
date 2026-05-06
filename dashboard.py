import re
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_EXCEL = Path("NeuroChain Survey  (Responses).xlsx")
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/u/1/d/"
    "1d7mIz29mjaeRL9sNCw-xn2Jnyscrgkt_EET33RdOiYI/edit"
    "?resourcekey&usp=forms_web_b&urp=linked#gid=2039873207"
)

RESPONDENT_COL = "Respondent Type"
STATUS_COL = "Submission Status"
TIMESTAMP_COL = "Timestamp"

SECTION_KEYWORDS = {
    "Parent": [
        "child",
        "parent",
        "therapy",
        "specialist sooner",
        "home video",
        "fraction of the cost",
        "child's progress",
    ],
    "Professional": [
        "professional role",
        "practice",
        "clinical",
        "patient",
        "client",
        "clinic",
        "documentation",
        "dashboard",
    ],
    "Teacher": [
        "teacher",
        "school",
        "student",
        "class",
        "classroom",
        "parents of children",
    ],
}


st.set_page_config(
    page_title="NeuroChain Survey Dashboard",
    page_icon="",
    layout="wide",
)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df.fillna("")


def google_sheet_export_url(url: str) -> str:
    parsed = urlparse(url)
    match = re.search(r"/spreadsheets/(?:u/\d+/)?d/([^/]+)", parsed.path)
    if not match:
        raise ValueError("This does not look like a Google Sheets URL.")

    sheet_id = match.group(1)
    gid = parse_qs(parsed.fragment).get("gid", ["0"])[0]
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


@st.cache_data(show_spinner=False)
def load_local_excel(path: str) -> pd.DataFrame:
    return clean_columns(pd.read_excel(path, dtype=str))


@st.cache_data(show_spinner=False)
def load_uploaded_excel(content: bytes) -> pd.DataFrame:
    return clean_columns(pd.read_excel(BytesIO(content), dtype=str))


@st.cache_data(show_spinner=False)
def load_google_sheet(url: str) -> pd.DataFrame:
    csv_url = google_sheet_export_url(url)
    return clean_columns(pd.read_csv(csv_url, dtype=str))


def short_label(label: str, max_len: int = 82) -> str:
    label = " ".join(str(label).split())
    return label if len(label) <= max_len else label[: max_len - 1].rstrip() + "..."


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", pd.NA), errors="coerce")


def likely_scale_columns(df: pd.DataFrame) -> list[str]:
    scale_cols = []
    for col in df.columns:
        values = numeric_series(df[col]).dropna()
        if values.empty:
            continue
        unique = set(values.astype(int).astype(str))
        in_range = values.between(1, 5).mean() >= 0.95
        looks_named = bool(re.search(r"\[1-|1[- ]|5[- ]|unlikely|likely|useful|valuable", col, re.I))
        enough = len(values) >= max(3, int(len(df) * 0.03))
        if in_range and enough and (looks_named or unique.issubset({"1", "2", "3", "4", "5"})):
            scale_cols.append(col)
    return scale_cols


def categorical_columns(df: pd.DataFrame) -> list[str]:
    ignored = {TIMESTAMP_COL, "Email address", "Email", "Submission Note", "Submitted At"}
    cols = []
    for col in df.columns:
        if col in ignored:
            continue
        non_empty = df[col].astype(str).str.strip().replace("", pd.NA).dropna()
        if non_empty.empty:
            continue
        unique_count = non_empty.nunique()
        if unique_count <= min(30, max(6, len(non_empty) * 0.7)):
            cols.append(col)
    return cols


def one_column_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    counts = (
        df[col]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .reset_index()
    )
    counts.columns = ["Answer", "Responses"]
    return counts


def multi_answer_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for value in df[col].astype(str).str.strip():
        if not value:
            continue
        parts = [part.strip() for part in value.split(",") if part.strip()]
        rows.extend(parts or [value])
    if not rows:
        return pd.DataFrame(columns=["Answer", "Responses"])
    return pd.Series(rows).value_counts().reset_index(name="Responses").rename(columns={"index": "Answer"})


def scale_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        values = numeric_series(df[col]).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "Question": short_label(col, 110),
                "Average": round(values.mean(), 2),
                "Responses": int(values.count()),
            }
        )
    return pd.DataFrame(rows).sort_values("Average", ascending=False)


def question_group(col: str) -> str:
    lowered = col.lower()
    for group, keywords in SECTION_KEYWORDS.items():
        if any(word in lowered for word in keywords):
            return group
    return "General"


def render_metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def load_data_panel() -> pd.DataFrame:
    st.sidebar.header("Data")
    source = st.sidebar.radio(
        "Source",
        ["Local Excel", "Upload Excel", "Google Sheets URL"],
        label_visibility="collapsed",
    )

    if source == "Upload Excel":
        uploaded = st.sidebar.file_uploader("Upload responses workbook", type=["xlsx", "xls"])
        if not uploaded:
            st.info("Upload the Google Form responses workbook to begin.")
            st.stop()
        return load_uploaded_excel(uploaded.getvalue())

    if source == "Google Sheets URL":
        url = st.sidebar.text_input("Sheet URL", value=DEFAULT_SHEET_URL)
        try:
            return load_google_sheet(url)
        except Exception as exc:
            st.error(
                "Could not read the Google Sheet as CSV. Make sure it is shared publicly, "
                "or use the local/upload option."
            )
            st.caption(str(exc))
            st.stop()

    if not DEFAULT_EXCEL.exists():
        st.error(f"Local file not found: {DEFAULT_EXCEL.resolve()}")
        st.stop()
    return load_local_excel(str(DEFAULT_EXCEL))


df = load_data_panel()

st.title("NeuroChain Survey Dashboard")
st.caption("Interactive view of Google Form responses, adoption signals, barriers, and stakeholder segments.")

if df.empty:
    st.warning("No rows found in the selected data source.")
    st.stop()

filtered = df.copy()

st.sidebar.header("Filters")
if RESPONDENT_COL in filtered.columns:
    respondent_options = sorted(
        [v for v in filtered[RESPONDENT_COL].astype(str).str.strip().unique() if v]
    )
    if respondent_options:  # Only show filter if there are options
        selected_respondents = st.sidebar.multiselect(
            "Respondent type",
            respondent_options,
            default=respondent_options,
        )
        if selected_respondents:
            filtered = filtered[filtered[RESPONDENT_COL].isin(selected_respondents)]

if STATUS_COL in filtered.columns:
    status_options = sorted([v for v in filtered[STATUS_COL].astype(str).str.strip().unique() if v])
    if status_options:
        selected_statuses = st.sidebar.multiselect("Submission status", status_options, default=status_options)
        if selected_statuses:
            filtered = filtered[filtered[STATUS_COL].isin(selected_statuses)]

search_text = st.sidebar.text_input("Find text in any answer").strip().lower()
if search_text:
    mask = filtered.astype(str).apply(
        lambda row: row.str.lower().str.contains(search_text, regex=False).any(),
        axis=1,
    )
    filtered = filtered[mask]

scale_cols = likely_scale_columns(filtered)
cat_cols = categorical_columns(filtered)

total_responses = len(filtered)
response_types = len([v for v in filtered[RESPONDENT_COL].unique() if str(v).strip()]) if RESPONDENT_COL in filtered.columns else 0
scale_count = len(scale_cols)
completion = ""
if STATUS_COL in filtered.columns and total_responses:
    submitted = filtered[STATUS_COL].astype(str).str.contains("submitted", case=False, na=False).sum()
    completion = f"{submitted / total_responses:.0%}"

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_metric("Responses", f"{total_responses:,}")
with kpi_cols[1]:
    render_metric("Segments", f"{response_types:,}")
with kpi_cols[2]:
    render_metric("Likert questions", f"{scale_count:,}")
with kpi_cols[3]:
    render_metric("Submitted", completion or "N/A")

tab_overview, tab_questions, tab_likert = st.tabs(
    ["Overview", "Question Explorer", "Likert Signals"]
)

with tab_overview:
    left, right = st.columns([1, 1])

    with left:
        if RESPONDENT_COL in filtered.columns:
            counts = one_column_counts(filtered, RESPONDENT_COL)
            fig = px.bar(
                counts,
                x="Responses",
                y="Answer",
                orientation="h",
                title="Responses by respondent type",
                color="Answer",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Responses")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if STATUS_COL in filtered.columns and filtered[STATUS_COL].astype(str).str.strip().any():
            status_counts = one_column_counts(filtered, STATUS_COL)
            fig = px.pie(
                status_counts,
                names="Answer",
                values="Responses",
                title="Submission status",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig, use_container_width=True)
        elif scale_cols:
            summary = scale_summary(filtered, scale_cols).head(8)
            fig = px.bar(
                summary,
                x="Average",
                y="Question",
                orientation="h",
                title="Top average 1-5 signals",
                color="Average",
                color_continuous_scale="Teal",
                range_x=[0, 5],
            )
            fig.update_layout(yaxis_title="", xaxis={"dtick": 1})
            st.plotly_chart(fig, use_container_width=True)

    if TIMESTAMP_COL in filtered.columns:
        timeline = filtered.copy()
        timeline[TIMESTAMP_COL] = pd.to_datetime(timeline[TIMESTAMP_COL], errors="coerce")
        timeline = timeline.dropna(subset=[TIMESTAMP_COL])
        if not timeline.empty:
            timeline["Date"] = timeline[TIMESTAMP_COL].dt.date
            group_cols = ["Date"]
            color = None
            if RESPONDENT_COL in timeline.columns:
                group_cols.append(RESPONDENT_COL)
                color = RESPONDENT_COL
            daily = timeline.groupby(group_cols).size().reset_index(name="Responses")
            fig = px.line(
                daily,
                x="Date",
                y="Responses",
                color=color,
                markers=True,
                title="Responses over time",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_questions:
    if not cat_cols:
        st.info("No categorical questions found in the filtered data.")
    else:
        default_question = RESPONDENT_COL if RESPONDENT_COL in cat_cols else cat_cols[0]
        question = st.selectbox(
            "Question",
            cat_cols,
            index=cat_cols.index(default_question),
            format_func=short_label,
        )
        split_multi = st.checkbox("Split comma-separated answers", value=True)
        
        st.subheader(f"Overall responses for: {short_label(question)}")
        counts = multi_answer_counts(filtered, question) if split_multi else one_column_counts(filtered, question)
        counts = counts.head(20)
        
        if counts.empty:
            st.info("No answers for this question after filters.")
        else:
            fig = px.bar(
                counts.sort_values("Responses"),
                x="Responses",
                y="Answer",
                orientation="h",
                title=f"Overall: {short_label(question, 100)}",
                color="Responses",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(yaxis_title="", xaxis_title="Responses", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if RESPONDENT_COL in filtered.columns and question != RESPONDENT_COL:
            st.subheader("Segment-by-Segment Breakdown")
            
            respondent_types = sorted([r for r in filtered[RESPONDENT_COL].unique() if r.strip()])
            
            for i in range(0, len(respondent_types), 2):
                col1, col2 = st.columns(2)
                
                # Segment 1
                with col1:
                    if i < len(respondent_types):
                        resp_type = respondent_types[i]
                        segment_data = filtered[filtered[RESPONDENT_COL] == resp_type]
                        
                        if not segment_data.empty:
                            segment_counts = multi_answer_counts(segment_data, question) if split_multi else one_column_counts(segment_data, question)
                            segment_counts = segment_counts.head(15)
                            
                            if not segment_counts.empty:
                                fig = px.bar(
                                    segment_counts.sort_values("Responses"),
                                    x="Responses",
                                    y="Answer",
                                    orientation="h",
                                    title=f"{resp_type}",
                                    color="Responses",
                                    color_continuous_scale="Cividis",
                                )
                                fig.update_layout(yaxis_title="", height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write(f"**{resp_type}**")
                                st.info("No answers for this question in this segment.")
                
                # Segment 2
                with col2:
                    if i + 1 < len(respondent_types):
                        resp_type = respondent_types[i+1]
                        segment_data = filtered[filtered[RESPONDENT_COL] == resp_type]
                        
                        if not segment_data.empty:
                            segment_counts = multi_answer_counts(segment_data, question) if split_multi else one_column_counts(segment_data, question)
                            segment_counts = segment_counts.head(15)
                            
                            if not segment_counts.empty:
                                fig = px.bar(
                                    segment_counts.sort_values("Responses"),
                                    x="Responses",
                                    y="Answer",
                                    orientation="h",
                                    title=f"{resp_type}",
                                    color="Responses",
                                    color_continuous_scale="Cividis",
                                )
                                fig.update_layout(yaxis_title="", height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write(f"**{resp_type}**")
                                st.info("No answers for this question in this segment.")

with tab_likert:
    if not scale_cols:
        st.info("No 1-5 scale questions found in the filtered data.")
    else:
        summary = scale_summary(filtered, scale_cols)
        group = st.radio(
            "Question group",
            ["All", "Parent", "Professional", "Teacher", "General"],
            horizontal=True,
            index=0,
        )
        if group != "All":
            allowed = [col for col in scale_cols if question_group(col) == group]
            summary = scale_summary(filtered, allowed) if allowed else pd.DataFrame()

        if summary.empty:
            st.info("No scale questions in this group for the current filters.")
        else:
            fig = px.bar(
                summary.sort_values("Average"),
                x="Average",
                y="Question",
                orientation="h",
                color="Average",
                color_continuous_scale="Tealgrn",
                range_x=[0, 5],
                title="Average score by 1-5 scale question",
                hover_data=["Responses"],
            )
            fig.update_layout(yaxis_title="", xaxis={"dtick": 1}, height=max(520, 28 * len(summary)))
            st.plotly_chart(fig, use_container_width=True)

            selected_scale = st.selectbox("Distribution", scale_cols, format_func=short_label)
            dist = numeric_series(filtered[selected_scale]).dropna().astype(int).value_counts().sort_index()
            dist_df = dist.reset_index()
            dist_df.columns = ["Score", "Responses"]
            fig = px.bar(
                dist_df,
                x="Score",
                y="Responses",
                title=short_label(selected_scale, 120),
                color="Score",
                color_continuous_scale="Teal",
            )
            fig.update_xaxes(dtick=1, range=[0.5, 5.5])
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced detailed analysis section
        st.subheader("Detailed Likert Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            if not summary.empty:
                # Score distribution across top questions
                top_qs = summary.head(10)
                st.write("**Top 10 Questions by Average Score**")
                st.dataframe(top_qs, use_container_width=True, hide_index=True)
        
        with analysis_col2:
            if not summary.empty:
                # Statistics summary
                stats_data = []
                for col in scale_cols[:15]:
                    values = numeric_series(filtered[col]).dropna()
                    if len(values) > 0:
                        try:
                            stats_data.append({
                                "Question": short_label(col, 50),
                                "Mean": round(values.mean(), 2),
                                "Median": int(values.median()),
                                "Std Dev": round(values.std(), 2),
                                "N": int(len(values)),
                            })
                        except Exception:
                            pass
                if stats_data:
                    st.write("**Statistical Summary**")
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        # Segment comparison heatmap
        if RESPONDENT_COL in filtered.columns and len(scale_cols) > 0:
            st.write("**Average Scores by Respondent Type**")
            heatmap_data = []
            respondent_types = sorted([r for r in filtered[RESPONDENT_COL].unique() if r.strip()])
            
            for resp_type in respondent_types:
                row_data = {"Respondent Type": resp_type}
                subset = filtered[filtered[RESPONDENT_COL] == resp_type]
                for col in scale_cols[:15]:
                    values = numeric_series(subset[col]).dropna()
                    if len(values) > 0:
                        row_data[short_label(col, 40)] = round(values.mean(), 2)
                heatmap_data.append(row_data)
            
            if heatmap_data and len(heatmap_data) > 0:
                heatmap_df = pd.DataFrame(heatmap_data).set_index("Respondent Type")
                # Remove rows and columns that are all NaN
                heatmap_df = heatmap_df.dropna(how='all').dropna(axis=1, how='all')
                
                if not heatmap_df.empty and len(heatmap_df.columns) > 0 and len(heatmap_df) > 0:
                    fig = px.imshow(
                        heatmap_df,
                        color_continuous_scale="RdYlGn",
                        range_color=[1, 5],
                        title="Heatmap: Average Scores by Segment",
                        aspect="auto",
                    )
                    fig.update_layout(height=250 + (50 * len(heatmap_df)))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data to display heatmap for this segment.")
        
        # Response rate analysis
        if scale_cols:
            st.write("**Response Rate by Question**")
            response_rates = []
            total_respondents = len(filtered)
            
            for col in scale_cols[:20]:
                try:
                    response_count = numeric_series(filtered[col]).dropna().count()
                    if response_count > 0:
                        rate = (response_count / total_respondents * 100) if total_respondents > 0 else 0
                        response_rates.append({
                            "Question": short_label(col, 60),
                            "Responses": int(response_count),
                            "Response Rate": f"{rate:.1f}%"
                        })
                except Exception:
                    pass
            
            if response_rates:
                rate_df = pd.DataFrame(response_rates).sort_values("Responses", ascending=False)
                fig = px.bar(
                    rate_df.head(15),
                    x="Responses",
                    y="Question",
                    orientation="h",
                    title="Top 15 Questions by Response Count",
                    color="Responses",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(yaxis_title="", xaxis_title="Number of Responses", height=max(400, 30*min(15, len(rate_df))))
                st.plotly_chart(fig, use_container_width=True)


