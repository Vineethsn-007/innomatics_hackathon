import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import io
import collections
from typing import Dict, List, Tuple

from omr_contour_mode import evaluate_contour_mode, SubjectNames
from load_answer_key import load_answer_key_from_sheet

st.set_page_config(layout="wide", page_title="OMR Grading System")
st.title("📄 Automated OMR Grading System")

with st.sidebar:
    st.header("📁 Upload Files")
    img_file = st.file_uploader("Upload OMR Sheet Image", type=["jpg","jpeg","png"])
    key_file = st.file_uploader("Upload Answer Key Excel", type=["xlsx"])
    sheet_name = st.selectbox("Select Answer Key Sheet", ["Set - A", "Set - B"])
    
    st.header("⚙️ Processing Parameters")
    rect_w = st.number_input("Rectified Width", value=1000, min_value=500, max_value=2000)
    rect_h = st.number_input("Rectified Height", value=1400, min_value=700, max_value=2800)
    bubble_thresh = st.slider("Bubble Fill Threshold", 0.0, 1.0, 0.16, 0.01,
                              help="Lower = more sensitive, Higher = less sensitive")
    margin = st.slider("Selection Margin", 0.0, 0.2, 0.05, 0.01,
                       help="Minimum difference between best and second-best bubble")
    
    st.header("🔧 Advanced Settings")
    show_debug = st.checkbox("Show Debug Visualization", value=True)
    show_detailed = st.checkbox("Show Detailed Analysis", value=False)

def grade_answers(student_answers: Dict[str, Dict[int, str]], answer_key: Dict[str, str]) -> Tuple[Dict[str, int], int, Dict[str, str]]:
    """Grade student answers against answer key"""
    scores = {subject: 0 for subject in SubjectNames}
    total_score = 0
    detailed_results = {}
    
    for subject in SubjectNames:
        for q_num in range(1, 21):  # Questions 1-20 for each subject
            key = f"{subject} Q{q_num}"
            student_ans = student_answers.get(subject, {}).get(q_num, "")
            correct_ans = answer_key.get(key, "")
            
            # Normalize answers for comparison
            student_ans_norm = student_ans.upper().strip()
            correct_ans_norm = correct_ans.upper().strip()
            
            is_correct = False
            if student_ans_norm and correct_ans_norm:
                # Handle multiple correct answers
                if "," in correct_ans_norm:
                    is_correct = student_ans_norm in correct_ans_norm.split(",")
                else:
                    is_correct = student_ans_norm == correct_ans_norm
            
            if is_correct:
                scores[subject] += 1
                total_score += 1
            
            # Store detailed result
            status = "✅ Correct" if is_correct else "❌ Wrong" if student_ans_norm else "⚠️ Unanswered"
            detailed_results[key] = {
                "subject": subject,
                "question": q_num,
                "student_answer": student_ans_norm or "Not answered",
                "correct_answer": correct_ans_norm or "N/A",
                "status": status,
                "is_correct": is_correct
            }
    
    return scores, total_score, detailed_results

def create_visualization(warped: np.ndarray, binary: np.ndarray, student_answers: Dict[str, Dict[int, str]]) -> np.ndarray:
    """Create visualization showing detected bubbles"""
    vis = warped.copy()
    h, w = vis.shape[:2]
    
    # Draw grid overlay to show detected regions
    for col in range(5):  # 5 subjects
        x = int((col + 0.5) * w / 5)
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
    
    for row in range(20):  # 20 questions per subject
        y = int((row + 0.5) * h / 20)
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
    
    # Add subject labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, subject in enumerate(SubjectNames):
        x = int((i + 0.5) * w / 5)
        cv2.putText(vis, subject[:8], (x-40, 30), font, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, subject[:8], (x-40, 30), font, 0.5, (0, 0, 255), 1)
    
    return vis

# Main content area
if img_file and key_file:
    try:
        # Load and prepare image
        with st.spinner("Loading image..."):
            image = Image.open(img_file).convert("RGB")
            img_array = np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        
        # Load answer key
        with st.spinner("Loading answer key..."):
            key_bytes = io.BytesIO(key_file.getbuffer())
            answer_key = load_answer_key_from_sheet(key_bytes, sheet_name=sheet_name)
            st.sidebar.success(f"✅ Loaded {len(answer_key)} answers")
        
        # Process OMR sheet
        with st.spinner("Processing OMR sheet..."):
            student_answers, _, warped, binary = evaluate_contour_mode(
                img_array, 
                warp_w=rect_w, 
                warp_h=rect_h, 
                threshold=bubble_thresh,
                margin=margin
            )
            
            # Grade the answers
            scores, total_score, detailed_results = grade_answers(student_answers, answer_key)
            
            # Create visualization
            vis = create_visualization(warped, binary, student_answers)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Original OMR Sheet")
            st.image(warped[:, :, ::-1], use_container_width=True)
            
            if show_debug:
                st.subheader("🔍 Binary Image")
                st.image(binary, use_container_width=True)
        
        with col2:
            st.subheader("✅ Processing Visualization")
            st.image(vis[:, :, ::-1], use_container_width=True)
            
            st.subheader("📝 Detected Answers Summary")
            for subject in SubjectNames:
                answers = student_answers.get(subject, {})
                answered_count = sum(1 for ans in answers.values() if ans)
                st.write(f"**{subject}**: {answered_count}/20 answered")
        
        # Display scores
        st.subheader("📊 Evaluation Results")
        
        # Score summary
        col_score1, col_score2, col_score3 = st.columns(3)
        with col_score1:
            st.metric("Total Score", f"{total_score}/100", f"{total_score}%")
        with col_score2:
            total_answered = sum(sum(1 for ans in subj_answers.values() if ans) 
                               for subj_answers in student_answers.values())
            st.metric("Questions Answered", f"{total_answered}/100")
        with col_score3:
            percentage = (total_score / 100) * 100 if total_score > 0 else 0
            st.metric("Percentage", f"{percentage:.1f}%")
        
        # Subject-wise scores
        st.subheader("📚 Subject-wise Performance")
        score_data = []
        for subject in SubjectNames:
            score = scores.get(subject, 0)
            answered = sum(1 for ans in student_answers.get(subject, {}).values() if ans)
            score_data.append({
                "Subject": subject,
                "Score": score,
                "Out of": 20,
                "Answered": answered,
                "Percentage": (score / 20 * 100) if score > 0 else 0
            })
        
        score_df = pd.DataFrame(score_data)
        
        # Display as a nice table
        st.dataframe(
            score_df.style.format({
                "Score": "{:.0f}",
                "Out of": "{:.0f}",
                "Answered": "{:.0f}",
                "Percentage": "{:.1f}%"
            }).background_gradient(subset=["Percentage"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True
        )
        
        # Detailed answer analysis
        if show_detailed:
            with st.expander("📝 Detailed Answer Analysis", expanded=True):
                # Create detailed dataframe
                detailed_data = []
                for key, result in detailed_results.items():
                    detailed_data.append({
                        "Subject": result["subject"],
                        "Question": result["question"],
                        "Student Answer": result["student_answer"],
                        "Correct Answer": result["correct_answer"],
                        "Status": result["status"]
                    })
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df = detailed_df.sort_values(["Subject", "Question"])
                
                # Filter options
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    subject_filter = st.multiselect(
                        "Filter by Subject",
                        options=SubjectNames,
                        default=SubjectNames
                    )
                with filter_col2:
                    status_filter = st.multiselect(
                        "Filter by Status",
                        options=["✅ Correct", "❌ Wrong", "⚠️ Unanswered"],
                        default=["✅ Correct", "❌ Wrong", "⚠️ Unanswered"]
                    )
                
                filtered_df = detailed_df[
                    (detailed_df["Subject"].isin(subject_filter)) &
                    (detailed_df["Status"].isin(status_filter))
                ]
                
                st.dataframe(filtered_df, use_container_width=True)
        
        st.subheader("📥 Download Results")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = {
                "Metric": ["Total Score", "Percentage", "Questions Answered", "Questions Unanswered"],
                "Value": [
                    f"{total_score}/100",
                    f"{percentage:.1f}%",
                    total_answered,
                    100 - total_answered
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            # Subject scores
            score_df.to_excel(writer, sheet_name="Subject Scores", index=False)
            
            # Detailed answers
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name="Detailed Answers", index=False)
            
            # Raw detected answers
            raw_data = []
            for subject in SubjectNames:
                for q_num in range(1, 21):
                    ans = student_answers.get(subject, {}).get(q_num, "")
                    raw_data.append({
                        "Subject": subject,
                        "Question": q_num,
                        "Detected Answer": ans or "Not detected"
                    })
            pd.DataFrame(raw_data).to_excel(writer, sheet_name="Raw Detection", index=False)
            
        output.seek(0)
        
        st.download_button(
            label="📊 Download Complete Report (Excel)",
            data=output,
            file_name=f"omr_grading_report_{sheet_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"❌ Error processing OMR sheet: {str(e)}")
        st.info("Tips:\n- Ensure the image is clear and well-lit\n- Try adjusting the threshold values\n- Make sure the answer key format is correct")
        
        with st.expander("🔧 Debug Information"):
            st.write("**Error Details:**")
            st.code(str(e))
            if img_file:
                st.write(f"**Image Info:** {img_file.name}, {img_file.size} bytes")
            if key_file:
                st.write(f"**Answer Key Info:** {key_file.name}, {key_file.size} bytes")
        
else:
    # Instructions when no files are uploaded
    st.info("👈 Please upload both an OMR sheet image and answer key Excel file to begin.")
    
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use this OMR Grading System:
        
        1. **Upload OMR Sheet Image**: Take a clear photo or scan of the filled OMR sheet
        2. **Upload Answer Key**: Excel file with sheets named "Set - A" or "Set - B"
        3. **Select the correct answer sheet** from the dropdown
        4. **Adjust parameters** if needed:
           - **Bubble Fill Threshold**: Controls sensitivity of bubble detection
           - **Selection Margin**: Minimum difference between best and second-best bubble
           - **Rectified dimensions**: Size of the processed image
        5. **View results** and download the grading report
        
        ### Answer Key Format:
        The Excel file should have columns for each subject with entries like:
        - `1 - a` or `1. a` for single answer
        - `16 - a,b,c,d` for multiple correct answers
        
        ### Supported Subjects:
        - PYTHON (Questions 1-20)
        - DATA ANALYSIS (Questions 21-40) 
        - MySQL (Questions 41-60)
        - POWER BI (Questions 61-80)
        - Adv STATS (Questions 81-100)
        
        ### Troubleshooting:
        - If bubbles are not detected properly, try adjusting the threshold
        - Ensure the image is not tilted or distorted
        - Check that the answer key format matches the expected pattern
        - Use the debug visualization to see what bubbles were detected
        """)
    
    # Sample data display
    with st.expander("📊 Sample Answer Key Format"):
        sample_data = {
            "PYTHON": ["1 - a", "2 - c", "3 - b", "4 - d"],
            "EDA": ["21 - a", "22 - d", "23 - b", "24 - a"],
            "SQL": ["41 - b", "42 - c", "43 - d", "44 - a"],
            "POWER BI": ["61 - a", "62 - b", "63 - c", "64 - d"],
            "SATISTICS": ["81 - d", "82 - a", "83 - b", "84 - c"]
        }
        st.dataframe(pd.DataFrame(sample_data))

st.markdown("---")
st.markdown("**OMR Grading System** - Automated bubble sheet evaluation using computer vision")
