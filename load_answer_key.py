import pandas as pd
import re

def load_answer_key_from_sheet(excel_file, sheet_name="Set - A"):
    """
    Reads answer key from Excel where cells are like:
    '1 - a' or '16 - a,b,c,d' or '81. a'
    Returns dict {"Python_Q1": "A", "Python_Q16": "A,B,C,D", ...}
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    answer_key = {}

    for col in df.columns:
        subject = str(col).strip()
        for cell in df[col].dropna():
            cell = str(cell).strip()

            # Match "QNo - answers" OR "QNo. answers"
            match = re.match(r"(\d+)\s*[\-\.]\s*(.+)", cell)
            if match:
                qno = match.group(1)
                ans = match.group(2).strip().upper()
                # Normalize multiple answers (a,b,c,d -> A,B,C,D)
                ans = ",".join([a.strip().upper() for a in ans.split(",")])
                key = f"{subject}_Q{qno}"
                answer_key[key] = ans
            else:
                print(f"Skipping cell {cell}: format not recognized")

    print(f"✅ Loaded {len(answer_key)} answers from {sheet_name}")
    return answer_key
