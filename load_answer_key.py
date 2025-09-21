import pandas as pd
import re

# Map spreadsheet headers to detector subject names
SUBJECT_MAP = {
    "PYTHON": "PYTHON",
    "EDA": "DATA ANALYSIS",
    "SQL": "MySQL", 
    "POWER BI": "POWER BI",
    "SATISTICS": "Adv STATS",   # Set-A typo: Satistics -> SATISTICS
    "STATISTICS": "Adv STATS",  # Set-B correct: Statistics -> STATISTICS
}

def load_answer_key_from_sheet(excel_file, sheet_name="Set - A"):
    """
    Reads answer key from Excel where cells are like:
      '1 - a' or '16 - a,b,c,d' or '81. a' or '21 : b'
    Returns dict like {"PYTHON Q1": "A", "DATA ANALYSIS Q21": "B", ...}
    Normalizes letters to uppercase and restricts to A-D.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    answer_key = {}

    for col in df.columns:
        raw_subject = str(col).strip()
        subject = SUBJECT_MAP.get(raw_subject.upper().strip(), raw_subject.upper().strip())

        for cell in df[col].dropna():
            s = str(cell).strip()
            # Support "QNo - answers" or "QNo. answers" or "QNo : answers"
            m = re.match(r"(\d+)\s*[\-\.]\s*(.+)", s)
            if not m:
                m = re.match(r"(\d+)\s*[:]\s*(.+)", s)
            if not m:
                print(f"Skipping cell '{s}': format not recognized")
                continue

            qno = m.group(1)
            ans = m.group(2).strip()
            # Normalize multiple answers: split on comma or whitespace, keep A-D only
            parts = [a.strip().upper() for a in re.split(r"[,\s]+", ans) if a.strip()]
            parts = [p for p in parts if p in ("A","B","C","D")]
            ans_norm = ",".join(parts)
            key = f"{subject} Q{qno}"
            answer_key[key] = ans_norm

    return answer_key
