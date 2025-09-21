import streamlit as st

# MUST be the very first Streamlit command
st.set_page_config(layout="wide", page_title="OMR Grading System")

# Test OpenCV import
try:
    import cv2
    opencv_available = True
except ImportError as e:
    st.error(f"""
    ❌ **OpenCV Import Failed**
    
    Error: {str(e)}
    
    **Required files:**
    1. `requirements.txt` with `opencv-python-headless==4.8.1.78`
    2. `packages.txt` with system dependencies
    3. `runtime.txt` with `python-3.9`
    """)
    st.stop()

# Continue with other imports
import numpy as np
from PIL import Image
import pandas as pd
import io
import collections
from typing import Dict, List, Tuple

from omr_contour_mode import evaluate_contour_mode, SubjectNames
from load_answer_key import load_answer_key_from_sheet

# Main app title
st.title("📄 Automated OMR Grading System")

# Sidebar for controls
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

# Main processing logic
if img_file is not None and key_file is not None:
    try:
        # Load answer key
        answer_key = load_answer_key_from_sheet(key_file, sheet_name)
        st.success(f"✅ Loaded {len(answer_key)} answers from {sheet_name}")
        
        # Process image
        image = Image.open(img_file)
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process OMR
        answers, scores, warped_bgr, binary = evaluate_contour_mode(
            image_bgr, rect_w, rect_h, bubble_thresh, margin
        )
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Processing Results")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Original & Processed", "Detection Results", "Grading Report"])
            
            with tab1:
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    # FIXED: Remove use_container_width parameter for images
                    st.image(image, caption="Original OMR Sheet", width=400)
                with subcol2:
                    if binary is not None:
                        # FIXED: Remove use_container_width parameter
                        st.image(binary, caption="Processed Binary Image", width=400, channels="GRAY")
            
            with tab2:
                st.write("**Detected Answers:**")
                for subject in SubjectNames:
                    if subject in answers:
                        st.write(f"**{subject}:**")
                        subject_answers = []
                        for q_num in sorted(answers[subject].keys()):
                            ans = answers[subject][q_num]
                            subject_answers.append(f"Q{q_num}: '{ans}'")
                        
                        # Display in rows of 5
                        for i in range(0, len(subject_answers), 5):
                            st.write(" | ".join(subject_answers[i:i+5]))
            
            with tab3:
                # Grading logic
                total_correct = 0
                total_questions = 0
                subject_scores = {}
                
                for subject in SubjectNames:
                    correct = 0
                    total = 0
                    
                    if subject in answers:
                        for q_num in sorted(answers[subject].keys()):
                            total += 1
                            detected = answers[subject][q_num]
                            expected_key = f"{subject} Q{q_num}"
                            expected = answer_key.get(expected_key, "")
                            
                            if detected == expected:
                                correct += 1
                    
                    subject_scores[subject] = (correct, total, (correct/max(1,total))*100)
                    total_correct += correct
                    total_questions += total
                
                # Display scores
                final_percentage = (total_correct/max(1,total_questions))*100
                
                st.metric("Overall Score", f"{total_correct}/{total_questions}", f"{final_percentage:.1f}%")
                
                # Subject breakdown
                score_data = []
                for subject, (correct, total, percentage) in subject_scores.items():
                    score_data.append({
                        "Subject": subject,
                        "Score": correct,
                        "Out of": total,
                        "Percentage": percentage
                    })
                
                score_df = pd.DataFrame(score_data)
                
                # Format dataframe with conditional styling
                try:
                    import matplotlib
                    styled_df = score_df.style.format({
                        "Score": ".0f", 
                        "Out of": ".0f", 
                        "Percentage": ".1f"
                    }).background_gradient(subset="Percentage", cmap="RdYlGn", vmin=0, vmax=100)
                    # Use try/except for dataframe parameter compatibility
                    try:
                        st.dataframe(styled_df, use_container_width=True)
                    except:
                        st.dataframe(styled_df)
                except ImportError:
                    styled_df = score_df.style.format({
                        "Score": ".0f", 
                        "Out of": ".0f", 
                        "Percentage": ".1f"
                    })
                    try:
                        st.dataframe(styled_df, use_container_width=True)
                    except:
                        st.dataframe(styled_df)
        
        with col2:
            st.subheader("📈 Quick Stats")
            
            if subject_scores:
                best_subject = max(subject_scores.items(), key=lambda x: x[1][2])
                st.metric("Best Subject", best_subject[0], f"{best_subject[1][2]:.1f}%")
                
                worst_subject = min(subject_scores.items(), key=lambda x: x[1][2])
                st.metric("Needs Improvement", worst_subject[0], f"{worst_subject[1][2]:.1f}%")
            
            # Download results
            if st.button("📥 Download Results"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    score_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=output.getvalue(),
                    file_name="omr_grading_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    except Exception as e:
        st.error(f"❌ Error processing OMR sheet: {str(e)}")
        st.info("Please check your files and try again.")

else:
    st.info("👆 Please upload both an OMR sheet image and an answer key Excel file to begin processing.")
    
    # Show sample files info
    with st.expander("📋 Sample Files & Instructions"):
        st.markdown("""
        **Required Files:**
        1. **OMR Sheet Image** (JPG/PNG) - Scanned answer sheet
        2. **Answer Key Excel** (XLSX) - With sheets named "Set - A" or "Set - B"
        
        **Supported OMR Format:**
        - 5 subjects × 20 questions each = 100 total questions
        - Subjects: PYTHON, DATA ANALYSIS, MySQL, POWER BI, Adv STATS
        - 4 options per question (A, B, C, D)
        
        **Tips for Best Results:**
        - Use good lighting when scanning
        - Ensure bubbles are completely filled
        - Keep sheets flat and aligned
        - Adjust threshold if detection seems too sensitive/insensitive
        """)
