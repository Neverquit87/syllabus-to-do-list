# Student_todolist_maker.py
import streamlit as st
import pdfplumber
import re
from hashlib import md5
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta, timezone
from io import BytesIO
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import requests
from collections import defaultdict

# -------------------------------------------------
# Streamlit config MUST be first Streamlit call
# -------------------------------------------------
st.set_page_config(page_title="Syllabus To-Do List", page_icon="📓")

# -------------------------------------------------
# Button styling (blue calendar, red PDF) + Canvas fetch button styling
# -------------------------------------------------
st.markdown(
    """
<style>

/* BLUE button (Add to calendar) */
div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
    background-color: #1976d2 !important;
    color: white !important;
    border: none !important;
}

/* RED button (Download as PDF) */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
    background-color: #d32f2f !important;
    color: white !important;
    border: none !important;
}

/* Remove ALL default borders, outlines, and focus rings */
button {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 10px !important;
}

/* Optional: hover effects */
div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:hover {
    background-color: #1565c0 !important;
}

div[data-testid="stHorizontalBlock"] > div:nth-child(2) button:hover {
    background-color: #b71c1c !important;
}

/* Make the Canvas fetch button look like a real rounded button */
div[data-testid="stButton"] button {
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    background: #f5f5f5 !important;
    color: #111 !important;
}
div[data-testid="stButton"] button:hover {
    background: #e9e9e9 !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Header UI
# -------------------------------------------------
st.title("Syllabus To-Do List Maker")
st.write("Upload your syllabus (PDF) to extract assignments from the Class/Course Schedule.")

uploaded_file = st.file_uploader("Upload Syllabus", type="pdf")

input_mode = st.radio(
    "Choose input method:",
    ["Upload syllabus PDF", "Manual paste list", "Import from Canvas"],
)

# -------------------------------------------------
# Keywords / filters
# -------------------------------------------------
TASK_KEYWORDS = [
    "quiz", "quizzes", "exam", "exams", "midterm", "final",
    "homework", "hw", "project", "proposal",
    "presentation", "discussion", "post", "reply", "lab", "activity",
    "cea", "srda", "due",
    "lecture video quiz questions",
    "all response comments",
    "icebreaker",
]

REJECT_CONTAINS = [
    "opens", "open at", "closes", "close at", "review", "zoom", "time tbd",
    "study for", "withdraw", "holiday", "spring break", "no class",
    "academic calendar", "office hours", "instructor", "contact", "attendance",
    "make-up", "make up", "policy", "penalty", "late submissions",
]

# -------------------------------------------------
# Regex helpers
# -------------------------------------------------
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$", re.IGNORECASE)

HEADER_FOOTER_RE = re.compile(
    r"(syllabus\s+spring|syllabus\s+fall|syllabus\s+summer|biol\s*\d{4}|clemson|understanding\s+animal\s+biology)",
    re.IGNORECASE,
)

HEADER_LINE_RE = re.compile(r"^\s*(week|unit|module)\s*\d+\b", re.IGNORECASE)

MONTH_RE = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

SCHEDULE_DATE_ROW_RE = re.compile(
    rf"^\s*(?P<dow>Mon|Tue|Tues|Wed|Thu|Thur|Fri|Sat|Sun)\.?,?\s+"
    rf"(?P<month>{MONTH_RE})\.?\s+"
    rf"(?P<day>\d{{1,2}})\b",
    re.IGNORECASE,
)

DAY_MONTH_DAY_ANY_RE = re.compile(
    rf"\b(?:Mon|Tue|Tues|Wed|Thu|Thur|Fri|Sat|Sun)\.?,?\s+"
    rf"(?P<month>{MONTH_RE})\.?\s+(?P<day>\d{{1,2}})\b",
    re.IGNORECASE,
)

MONTH_DAY_ANY_RE = re.compile(
    rf"\b(?P<month>{MONTH_RE})\.?\s+(?P<day>\d{{1,2}})\b",
    re.IGNORECASE,
)

DAY_DATE_PREFIX_RE = re.compile(
    rf"^\s*(Mon|Tue|Tues|Wed|Thu|Thur|Fri|Sat|Sun)\.?,?\s+{MONTH_RE}\.?\s+\d{{1,2}}\s+",
    re.IGNORECASE,
)

LECTURE_LINE_RE = re.compile(r"^\s*L\d+\s*:", re.IGNORECASE)
EXAM_ANCHOR_RE = re.compile(r"\b(midterm\s+exam|final\s+exam|exam)\b", re.IGNORECASE)

TASK_ANCHOR_SPLIT_RE = re.compile(
    r"(?="
    r"\bIcebreaker\b"
    r"|\bQuiz\s+\d+\b"
    r"|\bSRDA\b"
    r"|\bCEA\b"
    r"|\bMidterm\s+Exam\b"
    r"|\bFinal\s+Exam\b"
    r"|\bPresentation\s+Proposal\b"
    r"|\bAmazing\s+Animals\b"
    r"|\bAll\s+response\s+comments\b"
    r"|\blecture\s+video\s+quiz\s+questions\b"
    r")",
    re.IGNORECASE,
)

ADMIN_TAIL_RE = re.compile(
    r"\b(SPRING BREAK|LAST DAY TO WITHDRAW|MARTIN LUTHER KING)\b.*$",
    re.IGNORECASE,
)

# wrapped-line merging helpers
LAST_NAMES_INCOMPLETE_RE = re.compile(r"\(.*last names\b", re.IGNORECASE)
LETTER_RANGE_RE = re.compile(r"\b[A-Z]\s*-\s*[A-Z]\b")
LECTURE_RANGE_ONLY_RE = re.compile(r"^\s*L\d+\s*-\s*L?\d+\b", re.IGNORECASE)

# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def stable_id(text: str) -> str:
    return md5(text.encode("utf-8")).hexdigest()[:12]

def normalize_month(m: str) -> str:
    s = m.strip().lower().replace(".", "")
    if s.startswith("sept"):
        return "Sep"
    return s[:3].title()

def extract_text_from_pdf(file) -> str:
    pages = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)
    return "\n".join(pages)

def slice_to_schedule(all_lines: List[str]) -> List[str]:
    start_idx = None
    for i, line in enumerate(all_lines):
        if not line:
            continue
        ll = line.lower()
        if "class schedule" in ll or "course schedule" in ll:
            start_idx = i
            break
    return all_lines if start_idx is None else all_lines[start_idx:]

def contains_task_keyword(s: str) -> bool:
    ll = s.lower()
    return any(k in ll for k in TASK_KEYWORDS)

def should_reject(s: str) -> bool:
    ll = s.lower()
    return any(bad in ll for bad in REJECT_CONTAINS)

# -------------------------------------------------
# Manual paste parsing helpers
# -------------------------------------------------
MANUAL_DATE_PATTERNS = [
    re.compile(
        r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(?P<day>\d{1,2})\b",
        re.I,
    ),
    re.compile(r"\b(?P<m>\d{1,2})/(?P<d>\d{1,2})(?:/(?P<y>\d{2,4}))?\b"),
]

def normalize_manual_date(text: str) -> Optional[str]:
    t = text.strip()

    m = MANUAL_DATE_PATTERNS[0].search(t)
    if m:
        month = normalize_month(m.group("month"))
        day = int(m.group("day"))
        return f"{month} {day}"

    m2 = MANUAL_DATE_PATTERNS[1].search(t)
    if m2:
        mm = int(m2.group("m"))
        dd = int(m2.group("d"))
        month_map = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{month_map[mm]} {dd}"

    return None

def build_tasks_from_manual(text_block: str) -> List[Dict]:
    labels: List[str] = []
    for raw in text_block.splitlines():
        line = raw.strip()
        if not line:
            continue

        # allow bullets like "-", "•", "■"
        line = re.sub(r"^[\-\u2022\u25A0\*\s]+", "", line).strip()
        if not line:
            continue

        date_str = normalize_manual_date(line)
        if not date_str:
            continue

        cleaned = line.replace("—", "-").replace("–", "-")

        label_only = cleaned
        label_only = MANUAL_DATE_PATTERNS[0].sub("", label_only)
        label_only = MANUAL_DATE_PATTERNS[1].sub("", label_only)
        label_only = label_only.strip(" -:;—").strip()

        if not label_only:
            continue

        labels.append(f"{label_only} - {date_str}")

    seen = set()
    out = []
    for lbl in labels:
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)

    return [{"id": stable_id(lbl), "label": lbl} for lbl in out]

# -------------------------------------------------
# Canvas import
# -------------------------------------------------
def canvas_get_all(url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Fetch all pages from a Canvas endpoint."""
    out: List[dict] = []
    page = 1
    while True:
        p = {} if params is None else dict(params)
        p.update({"per_page": 100, "page": page})

        r = requests.get(url, headers=headers, params=p, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"Canvas API error {r.status_code}: {r.text}")

        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Canvas response (not a list): {data}")

        out.extend(data)
        if len(data) < 100:
            break
        page += 1

    return out

def fetch_canvas_tasks(domain: str, token: str, include_undated: bool = False) -> List[Dict]:
    """
    Pulls:
      - Assignments: /courses/:id/assignments
      - Quizzes:     /courses/:id/quizzes

    Notes:
      - Unpublished/locked items may not be returned by Canvas to student tokens.
      - If include_undated=False, we skip items with no due_at.
      - If include_undated=True, we include them as "(no due date)" and push them to the bottom in sorting.
    """
    domain = domain.strip().rstrip("/")
    headers = {"Authorization": f"Bearer {token.strip()}"}

    tasks: List[Dict] = []
    seen: set[Tuple[int, str]] = set()

    courses_url = f"{domain}/api/v1/users/self/courses"
    courses = canvas_get_all(courses_url, headers=headers, params={"enrollment_state": "active"})

    for course in courses:
        course_id = course.get("id")
        if not course_id:
            continue
        course_name = course.get("name") or f"Course {course_id}"

        # ---- Assignments ----
        assignments_url = f"{domain}/api/v1/courses/{course_id}/assignments"
        assignments = canvas_get_all(assignments_url, headers=headers)

        for a in assignments:
            name = a.get("name")
            due = a.get("due_at")  # ISO timestamp or None
            if not name:
                continue
            if not due and not include_undated:
                continue

            if due:
                due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                date_str = due_dt.strftime("%b %d").replace(" 0", " ")
                label = f"{name} - {date_str}"
                due_sort = due_dt
            else:
                label = f"{name} - (no due date)"
                date_str = ""
                due_sort = datetime.max.replace(tzinfo=timezone.utc)

            sig = (course_id, label)
            if sig in seen:
                continue
            seen.add(sig)

            tasks.append(
                {
                    "id": stable_id(f"{course_id}:{label}"),
                    "label": label,
                    "course_id": course_id,
                    "course_name": course_name,
                    "due_dt": due_sort,
                    "due_str": date_str,
                }
            )

           # ---- Quizzes ----
        quizzes_url = f"{domain}/api/v1/courses/{course_id}/quizzes"
        try:
            quizzes = canvas_get_all(quizzes_url, headers=headers)
        except RuntimeError as e:
            # Common if the Quizzes page is disabled for that course
            msg = str(e)
            if "disabled" in msg.lower() or "404" in msg:
                quizzes = []
            else:
                raise

        for q in quizzes:
            name = q.get("title")
            due = q.get("due_at")
            if not name:
                continue
            if not due and not include_undated:
                continue

            if due:
                due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                date_str = due_dt.strftime("%b %d").replace(" 0", " ")
                label = f"{name} - {date_str}"
                due_sort = due_dt
            else:
                label = f"{name} - (no due date)"
                date_str = ""
                due_sort = datetime.max.replace(tzinfo=timezone.utc)

            sig = (course_id, label)
            if sig in seen:
                continue
            seen.add(sig)

            tasks.append({
                "id": stable_id(f"{course_id}:{label}"),
                "label": label,
                "course_id": course_id,
                "course_name": course_name,
                "due_dt": due_sort,
                "due_str": date_str,
            })
    return tasks

# -------------------------------------------------
# Wrapped-line merge helpers
# -------------------------------------------------
def should_force_merge(prev: str, curr: str) -> bool:
    p = prev.strip()
    c = curr.strip()
    pl = p.lower()
    cl = c.lower()

    if "(" in p and ")" not in p:
        return True
    if LAST_NAMES_INCOMPLETE_RE.search(p) and not LETTER_RANGE_RE.search(p):
        return True
    if pl.endswith("for the") or pl.endswith("for"):
        return True
    if pl.startswith("all response comments") and "srda" not in pl and ("srda" in cl):
        return True
    if LECTURE_RANGE_ONLY_RE.match(p) and ("lecture video quiz questions" in cl):
        return True
    if p.endswith("-") or p.endswith("Last") or p.endswith("names"):
        return True

    return False

def merge_wrapped_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    for raw in lines:
        if not raw:
            continue
        line = raw.strip()
        if not line:
            continue

        if PAGE_NUM_RE.match(line):
            continue
        if HEADER_FOOTER_RE.search(line):
            continue
        if HEADER_LINE_RE.match(line):
            merged.append(line)
            continue

        if SCHEDULE_DATE_ROW_RE.match(line):
            merged.append(line)
            continue

        if merged and should_force_merge(merged[-1], line):
            merged[-1] += " " + line
            continue

        starts_like_task = bool(TASK_ANCHOR_SPLIT_RE.search(line))
        if not merged or starts_like_task:
            merged.append(line)
        else:
            merged[-1] += " " + line

    return merged

def clean_label(text: str) -> str:
    t = text.strip()

    t = re.sub(
        rf"\s*[—-]\s*(?:Mon|Tue|Tues|Wed|Thu|Thur|Fri|Sat|Sun)?\.?,?\s*{MONTH_RE}\.?\s+\d{{1,2}}\b.*$",
        "",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"\s*\bdue\b.*?\bby\b\s*\d{1,2}:\d{2}\s*(?:am|pm)?\b.*$",
        " due",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(r"\s{2,}", " ", t)
    return t.strip(" -:;—")

def extract_embedded_date(seg: str) -> Optional[str]:
    m = DAY_MONTH_DAY_ANY_RE.search(seg)
    if m:
        return f"{normalize_month(m.group('month'))} {m.group('day')}"
    m2 = MONTH_DAY_ANY_RE.search(seg)
    if m2:
        return f"{normalize_month(m2.group('month'))} {m2.group('day')}"
    return None

def strip_embedded_date_text(seg: str) -> str:
    seg = DAY_MONTH_DAY_ANY_RE.sub("", seg)
    seg = MONTH_DAY_ANY_RE.sub("", seg)
    seg = re.sub(r"\s{2,}", " ", seg)
    return seg.strip()

def is_real_task_label(label: str) -> bool:
    ll = label.lower()

    if "final exam" in ll:
        return False
    if LECTURE_LINE_RE.match(label) and "due" not in ll:
        return False
    if ll in {"due", "srda due", "quiz due", "assignment due"}:
        return False

    starts_ok = bool(
        re.match(
            r"^(icebreaker|quiz\s*\d+|srda|cea|midterm\s+exam|all response comments|lecture video quiz questions|amazing animals|presentation proposal|l\d+\s*-\s*l?\d+)",
            ll,
        )
    )
    return starts_ok

# -------------------------------------------------
# Core parsing (PDF schedule style)
# -------------------------------------------------
def parse_schedule_tasks(lines: List[str]) -> List[str]:
    tasks: List[str] = []
    seen: set[Tuple[str, str]] = set()
    current_date: Optional[str] = None

    pending_payload: Optional[str] = None
    pending_date: Optional[str] = None

    def add_task(label: str, date_str: str):
        label = label.strip()
        if not (6 <= len(label) <= 220):
            return
        if not is_real_task_label(label):
            return
        key = (label.lower(), date_str)
        if key in seen:
            return
        seen.add(key)
        tasks.append(f"{label} - {date_str}")

    def process_payload(payload: str, row_date: str):
        payload = ADMIN_TAIL_RE.sub("", payload).strip()
        row_override_date = extract_embedded_date(payload)

        for seg in [s.strip() for s in payload.split(";") if s.strip()]:
            seg = ADMIN_TAIL_RE.sub("", seg).strip()
            if not seg:
                continue
            if should_reject(seg) or not contains_task_keyword(seg):
                continue

            parts = [p.strip() for p in TASK_ANCHOR_SPLIT_RE.split(seg) if p.strip()]

            stitched: List[str] = []
            i = 0
            while i < len(parts):
                cur = parts[i].strip()
                nxt = parts[i + 1].strip() if i + 1 < len(parts) else ""
                cur_low = cur.lower()
                nxt_low = nxt.lower()

                if cur_low.endswith("for the") and nxt_low.startswith("srda"):
                    stitched.append(f"{cur} {nxt}".strip())
                    i += 2
                    continue

                if re.match(r"^\s*L\d+\s*-\s*L?\d+\s*$", cur, re.IGNORECASE) and (
                    "lecture video quiz questions" in nxt_low
                ):
                    stitched.append(f"{cur} {nxt}".strip())
                    i += 2
                    continue

                stitched.append(cur)
                i += 1

            for part in stitched:
                part = ADMIN_TAIL_RE.sub("", part).strip()
                if not part:
                    continue
                if should_reject(part) or not contains_task_keyword(part):
                    continue

                part_date = extract_embedded_date(part)
                date_for_this = part_date or row_override_date or row_date

                m_exam = EXAM_ANCHOR_RE.search(part)
                if m_exam:
                    part = part[m_exam.start():].strip()

                part = strip_embedded_date_text(part)
                label = clean_label(part)

                if row_override_date:
                    ll = label.lower()
                    if ll.startswith("srda") or "proposal" in ll or "amazing animals" in ll:
                        date_for_this = row_override_date

                add_task(label, date_for_this)

    def has_any_date_text(s: str) -> bool:
        return bool(extract_embedded_date(s))

    for raw in lines:
        if not raw:
            continue
        raw = raw.strip()
        if not raw:
            continue

        if PAGE_NUM_RE.match(raw):
            continue
        if HEADER_FOOTER_RE.search(raw):
            continue
        if HEADER_LINE_RE.match(raw):
            continue

        norm = raw.replace("–", "-").replace("—", "-").strip()

        m = SCHEDULE_DATE_ROW_RE.match(norm)
        if m:
            new_date = f"{normalize_month(m.group('month'))} {m.group('day')}"
            payload = SCHEDULE_DATE_ROW_RE.sub("", norm, count=1).strip(" -:;")

            if not payload and pending_payload:
                process_payload(pending_payload, new_date)
                pending_payload = None
                pending_date = None
                current_date = new_date
                continue

            if pending_payload and pending_date:
                process_payload(pending_payload, pending_date)
                pending_payload = None
                pending_date = None

            current_date = new_date

            if payload:
                process_payload(payload, current_date)
            continue

        if not current_date:
            continue

        if should_reject(norm) or not contains_task_keyword(norm):
            if pending_payload and pending_date:
                process_payload(pending_payload, pending_date)
                pending_payload = None
                pending_date = None
            continue

        norm_no_prefix = DAY_DATE_PREFIX_RE.sub("", norm).strip()

        if not has_any_date_text(norm_no_prefix):
            if pending_payload and pending_date:
                process_payload(pending_payload, pending_date)
            pending_payload = norm_no_prefix
            pending_date = current_date
            continue

        if pending_payload and pending_date:
            process_payload(pending_payload, pending_date)
            pending_payload = None
            pending_date = None

        process_payload(norm_no_prefix, current_date)

    if pending_payload and pending_date:
        process_payload(pending_payload, pending_date)

    return tasks

def parse_global_final_exam(all_lines: List[str]) -> List[str]:
    candidates: List[str] = []
    for raw in all_lines:
        if not raw:
            continue
        s = raw.strip()
        if not s:
            continue
        ll = s.lower()
        if "final exam" not in ll:
            continue
        if "due" not in ll:
            continue

        md = MONTH_DAY_ANY_RE.search(s)
        if not md:
            continue

        date_str = f"{normalize_month(md.group('month'))} {md.group('day')}"
        candidates.append(f"Final Exam due - {date_str}")

    return candidates[:1]

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def build_tasks(all_lines: List[str]) -> List[Dict]:
    schedule_lines = slice_to_schedule(all_lines)
    merged = merge_wrapped_lines(schedule_lines)

    schedule_tasks = parse_schedule_tasks(merged)
    final_exam_task = parse_global_final_exam(all_lines)

    labels = dedupe_preserve_order(schedule_tasks + final_exam_task)
    return [{"id": stable_id(lbl), "label": lbl} for lbl in labels]

# -------------------------------------------------
# Export helpers (ICS + PDF)
# -------------------------------------------------
def make_tasks_pdf(tasks: List[Dict], status: Dict[str, bool]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter

    left_margin = 0.75 * inch
    box_size = 10
    box_gap = 8

    y = height - 0.9 * inch
    line_height = 18

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, y, "Syllabus To-Do List")
    y -= 0.4 * inch

    c.setFont("Helvetica", 11)

    for t in tasks:
        checked = status.get(t["id"], False)

        if y < 0.75 * inch:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 0.9 * inch

        box_y = y - box_size + 3
        c.rect(left_margin, box_y, box_size, box_size)

        if checked:
            c.setLineWidth(2)
            c.line(left_margin + 2, box_y + 5, left_margin + 5, box_y + 2)
            c.line(left_margin + 5, box_y + 2, left_margin + box_size - 2, box_y + box_size - 2)

        text_x = left_margin + box_size + box_gap
        c.drawString(text_x, y - box_size + 3, t["label"])
        y -= line_height

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

def make_ics(tasks: List[Dict]) -> bytes:
    # Only export items with a real "Mon 1" style date at the end
    end_date_re = re.compile(rf"^(?P<mon>{MONTH_RE})\s+(?P<day>\d{{1,2}})$", re.IGNORECASE)

    def parse_date(date_str: str) -> datetime:
        year = 2026
        return datetime.strptime(f"{date_str} {year}", "%b %d %Y")

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Syllabus To-Do List Maker//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]

    dtstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for t in tasks:
        label = t["label"]
        if " - " not in label:
            continue

        title, date_str = label.rsplit(" - ", 1)
        date_str = date_str.strip()

        if not end_date_re.match(date_str):
            # skip "(no due date)" etc.
            continue

        try:
            d = parse_date(date_str)
        except Exception:
            continue

        dtstart = d.strftime("%Y%m%d")
        dtend = (d + timedelta(days=1)).strftime("%Y%m%d")
        uid = f"{uuid.uuid4()}@syllabus-todolist"

        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{dtstamp}",
                f"SUMMARY:{title.strip()}",
                f"DTSTART;VALUE=DATE:{dtstart}",
                f"DTEND;VALUE=DATE:{dtend}",
                "END:VEVENT",
            ]
        )

    lines.append("END:VCALENDAR")
    return "\r\n".join(lines).encode("utf-8")

# -------------------------------------------------
# UI logic
# -------------------------------------------------
tasks: List[Dict] = []

if input_mode == "Upload syllabus PDF":
    if uploaded_file is not None:
        full_text = extract_text_from_pdf(uploaded_file)
        all_lines = full_text.split("\n")
        tasks = build_tasks(all_lines)
    else:
        st.info("Upload a syllabus PDF to get started.")

elif input_mode == "Manual paste list":
    st.write("Paste tasks one-per-line. Include a due date like **Jan 18** or **3/15**.")
    manual_text = st.text_area(
        "Paste your tasks here:",
        height=220,
        placeholder="Quiz 1 - Jan 18\nSRDA Week 9 (Last names A-G) - Mar 15\nFinal Exam due - Apr 30",
    )
    if manual_text.strip():
        tasks = build_tasks_from_manual(manual_text)
    else:
        st.info("Paste a list with due dates to generate your checklist.")

elif input_mode == "Import from Canvas":
    st.write("Connect to Canvas to automatically import assignments + quizzes.")

    canvas_domain = st.text_input("Canvas domain", placeholder="https://clemson.instructure.com")
    canvas_token = st.text_input("Canvas API token", type="password")

    include_undated = st.checkbox("Include items without due dates (shows at bottom)", value=False)

    fetch_now = st.button("Fetch assignments from Canvas")

    if fetch_now:
        if not canvas_domain or not canvas_token:
            st.warning("Please enter both the Canvas domain and token.")
        else:
            with st.spinner("Fetching courses + assignments + quizzes from Canvas..."):
                try:
                    tasks = fetch_canvas_tasks(canvas_domain, canvas_token, include_undated=include_undated)
                    if not tasks:
                        st.warning(
                            "No items found. If your instructor hasn't published items yet (or they're locked), "
                            "Canvas may not return them. Also, many items may have no due dates."
                        )
                except Exception as e:
                    st.error(f"Canvas import failed: {e}")

# ---- Render results ----
if tasks:
    has_course = all(("course_name" in t and "due_dt" in t) for t in tasks)

    if has_course:
        grouped = defaultdict(list)
        for t in tasks:
            grouped[t["course_name"]].append(t)

        ids = [t["id"] for t in tasks]
        if "task_status" not in st.session_state:
            st.session_state["task_status"] = {}
        for tid in ids:
            st.session_state["task_status"].setdefault(tid, False)

        st.subheader("Assignments (by course)")

        total = 0
        done = 0

        for course_name in sorted(grouped.keys()):
            course_tasks = sorted(grouped[course_name], key=lambda x: x["due_dt"])

            with st.expander(course_name, expanded=True):
                for t in course_tasks:
                    total += 1
                    checked = st.checkbox(
                        t["label"],
                        value=st.session_state["task_status"].get(t["id"], False),
                        key=t["id"],
                    )
                    st.session_state["task_status"][t["id"]] = checked
                    if checked:
                        done += 1

        st.write(f"**Progress** {done} / {total} tasks completed")

    else:
        labels = [t["label"] for t in tasks]
        if "last_labels" not in st.session_state or st.session_state["last_labels"] != labels:
            st.session_state["task_status"] = {t["id"]: False for t in tasks}
            st.session_state["last_labels"] = labels

        st.subheader("Extracted Tasks")
        for t in tasks:
            st.session_state["task_status"][t["id"]] = st.checkbox(
                t["label"],
                value=st.session_state["task_status"].get(t["id"], False),
                key=t["id"],
            )

        completed = sum(1 for v in st.session_state["task_status"].values() if v)
        st.write(f"**Progress** {completed} / {len(tasks)} tasks completed")

    # ---- Exports ----
    pdf_bytes = make_tasks_pdf(tasks, st.session_state["task_status"])
    ics_bytes = make_ics(tasks)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Add to calendar",
            data=ics_bytes,
            file_name="syllabus_calendar.ics",
            mime="text/calendar",
            use_container_width=True,
            type="primary",
        )
    with col2:
        st.download_button(
            "Download as PDF",
            data=pdf_bytes,
            file_name="syllabus_todo_list.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

else:
    if input_mode == "Upload syllabus PDF" and uploaded_file is not None:
        st.warning("No clear tasks detected. Try the Manual paste option if your syllabus has no schedule.")