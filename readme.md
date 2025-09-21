## Automated OMR Evaluation

### Setup
pip install -r requirements.txt

### Files
- answer_key.xlsx → Your Excel with Set - A / Set - B
- student_omr.jpg → Example OMR sheet
- omr_processing.py → OMR pipeline
- roi_generator.py → ROI generator (tune margins to your sheet)
- load_answer_key.py → Loads answer keys from your Excel
- evaluate_cli.py → CLI runner
- app.py → Streamlit app

### Run CLI
python evaluate_cli.py --image student_omr.jpg --key answer_key.xlsx --set "Set - A"

### Run Web UI
streamlit run app.py
