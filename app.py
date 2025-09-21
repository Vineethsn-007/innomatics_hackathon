import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import io
from omr_processing import preprocess_sheet_from_array, evaluate_sheet, visualize_detection
from roi_generator import generate_rois
from load_answer_key import load_answer_key_from_sheet

st.set_page_config(layout="wide", page_title="OMR Evaluation System")
st.title("📄 Automated OMR Evaluation System")

# Sidebar controls
with st.sidebar:
    st.header("📁 Upload Files")
    img_file = st.file_uploader("Upload OMR Sheet Image", type=["jpg","jpeg","png"])
    key_file = st.file_uploader("Upload Answer Key Excel", type=["xlsx"])
    sheet_name = st.selectbox("Select Answer Key Sheet", ["Set - A", "Set - B"])
    
    st.header("⚙️ Processing Parameters")
    rect_w = st.number_input("Rectified Width", value=1000, min_value=500, max_value=2000)
    rect_h = st.number_input("Rectified Height", value=1400, min_value=700, max_value=2800)
    bubble_thresh = st.slider("Bubble Fill Threshold", 0.0, 1.0, 0.30, 0.05,
                              help="Lower = more sensitive, Higher = less sensitive")
    
    st.header("🔧 Fine-tuning Controls")
    with st.expander("Advanced Settings"):
        use_calibrated = st.checkbox("Use Calibrated ROIs", value=False,
                                    help="Use percentage-based ROI positions")
        show_binary = st.checkbox("Show Binary Image", value=False)
        save_debug = st.checkbox("Save Debug Images", value=False)

# Main content area
if img_file and key_file:
    try:
        # Load and prepare image
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        h_img, w_img = img_array.shape[:2]
        
        # Use full image as corners if corner detection fails
        corners = np.float32([[0,0],[w_img-1,0],[w_img-1,h_img-1],[0,h_img-1]])
        
        # Load answer key
        with st.spinner("Loading answer key..."):
            key_bytes = io.BytesIO(key_file.getbuffer())
            answer_key = load_answer_key_from_sheet(key_bytes, sheet_name=sheet_name)
            st.sidebar.success(f"✅ Loaded {len(answer_key)} answers")
        
        # Generate ROIs
        if use_calibrated:
            from roi_generator import generate_rois_calibrated
            bubble_roi = generate_rois_calibrated(template_width=rect_w, template_height=rect_h)
        else:
            bubble_roi = generate_rois(template_width=rect_w, template_height=rect_h)
        
        # Process image
        with st.spinner("Processing OMR sheet..."):
            binary, warped = preprocess_sheet_from_array(img_array, corners, rect_w, rect_h)
            
            # Evaluate
            scores, total, student_answers, audit = evaluate_sheet(
                binary, bubble_roi, answer_key, bubble_threshold=bubble_thresh
            )
            
            # Generate visualization
            vis = visualize_detection(warped, binary, bubble_roi, bubble_threshold=bubble_thresh)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Original OMR Sheet")
            st.image(warped[:, :, ::-1], use_container_width=True)
            
            if show_binary:
                st.subheader("🔍 Binary Image")
                st.image(binary, use_container_width=True)
        
        with col2:
            st.subheader("✅ Detected Bubbles")
            st.image(vis[:, :, ::-1], use_container_width=True)
            
            # Save debug images if requested
            if save_debug:
                cv2.imwrite("debug_binary.jpg", binary)
                cv2.imwrite("debug_visualization.jpg", vis)
                st.info("Debug images saved: debug_binary.jpg, debug_visualization.jpg")
        
        # Display scores
        st.subheader("📊 Evaluation Results")
        
        # Score summary
        col_score1, col_score2, col_score3 = st.columns(3)
        with col_score1:
            st.metric("Total Score", f"{total}/100", f"{total}%")
        with col_score2:
            answered = sum(1 for ans in student_answers.values() if ans not in ["Unanswered", "Multiple Marks"])
            st.metric("Questions Answered", f"{answered}/100")
        with col_score3:
            multiple = sum(1 for ans in student_answers.values() if ans == "Multiple Marks")
            st.metric("Multiple Marks", multiple)
        
        # Subject-wise scores
        st.subheader("📚 Subject-wise Performance")
        score_df = pd.DataFrame(list(scores.items()), columns=["Subject", "Score"])
        score_df["Out of"] = 20
        score_df["Percentage"] = (score_df["Score"] / 20 * 100).round(1)
        
        # Display as a nice table
        st.dataframe(
            score_df.style.format({
                "Score": "{:.0f}",
                "Out of": "{:.0f}",
                "Percentage": "{:.1f}%"
            }).background_gradient(subset=["Percentage"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True
        )
        
        # Detailed answer analysis
        with st.expander("📝 Detailed Answer Analysis"):
            rows = []
            for q, ans in student_answers.items():
                subject, qnum = q.rsplit("_Q", 1)
                rows.append({
                    "Subject": subject,
                    "Question": int(qnum),
                    "Student Answer": ans,
                    "Correct Answer": answer_key.get(q, "N/A"),
                    "Status": "✅" if ans == answer_key.get(q) else "❌" if ans not in ["Unanswered", "Multiple Marks"] else "⚠️",
                    "Marked Options": ", ".join(audit.get(q, []))
                })
            
            df_detailed = pd.DataFrame(rows)
            df_detailed = df_detailed.sort_values(["Subject", "Question"])
            
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                subject_filter = st.multiselect(
                    "Filter by Subject",
                    options=df_detailed["Subject"].unique(),
                    default=df_detailed["Subject"].unique()
                )
            with filter_col2:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=["✅", "❌", "⚠️"],
                    default=["✅", "❌", "⚠️"]
                )
            
            filtered_df = df_detailed[
                (df_detailed["Subject"].isin(subject_filter)) &
                (df_detailed["Status"].isin(status_filter))
            ]
            
            st.dataframe(filtered_df, use_container_width=True)
        
        # Generate Excel report
        st.subheader("📥 Download Results")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = {
                "Metric": ["Total Score", "Percentage", "Questions Answered", "Questions Unanswered", "Multiple Marks"],
                "Value": [
                    f"{total}/100",
                    f"{total}%",
                    answered,
                    100 - answered - multiple,
                    multiple
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            # Subject scores
            score_df.to_excel(writer, sheet_name="Subject Scores", index=False)
            
            # Detailed answers
            df_detailed.to_excel(writer, sheet_name="Detailed Answers", index=False)
            
        output.seek(0)
        
        st.download_button(
            label="📊 Download Complete Report (Excel)",
            data=output,
            file_name=f"omr_evaluation_{sheet_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"❌ Error processing OMR sheet: {str(e)}")
        st.info("Tips:\n- Ensure the image is clear and well-lit\n- Try adjusting the threshold value\n- Make sure the answer key format is correct")
        
else:
    # Instructions when no files are uploaded
    st.info("👈 Please upload both an OMR sheet image and answer key Excel file to begin.")
    
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use this OMR Evaluation System:
        
        1. **Upload OMR Sheet Image**: Take a clear photo or scan of the filled OMR sheet
        2. **Upload Answer Key**: Excel file with sheets named "Set - A" or "Set - B"
        3. **Select the correct answer sheet** from the dropdown
        4. **Adjust parameters** if needed:
           - **Bubble Fill Threshold**: Controls sensitivity of bubble detection
           - **Rectified dimensions**: Size of the processed image
        5. **View results** and download the evaluation report
        
        ### Answer Key Format:
        The Excel file should have columns for each subject with entries like:
        - `1 - a` or `1. a` for single answer
        - `16 - a,b,c,d` for multiple correct answers
        
        ### Troubleshooting:
        - If bubbles are not detected properly, try adjusting the threshold
        - Ensure the image is not tilted or distorted
        - Check that the answer key format matches the expected pattern
        """)
    
    # Sample data display
    with st.expander("📊 Sample Answer Key Format"):
        sample_data = {
            "PYTHON": ["1 - a", "2 - c", "3 - b", "4 - d"],
            "DATA_ANALYSIS": ["21 - a", "22 - d", "23 - b", "24 - a"],
            "MySQL": ["41 - b", "42 - c", "43 - d", "44 - a"],
        }
        st.dataframe(pd.DataFrame(sample_data))