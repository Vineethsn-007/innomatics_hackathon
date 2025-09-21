# roi_generator.py
"""
Generate ROIs for a sheet with 5 subjects x 20 questions each.
These values are calibrated for your specific OMR template.
"""

def generate_rois(template_width=1000, template_height=1400):
    """
    Generate ROIs based on the actual OMR sheet layout.
    Returns: bubble_roi dict { "Subject": [(x,y,w,h,num_options), ...], ... }
    
    The layout from your image shows:
    - 5 columns: PYTHON, DATA ANALYSIS, MySQL, POWER BI, Adv STATS
    - 20 rows of questions
    - 4 bubbles per question (A, B, C, D)
    """
    
    # These measurements are based on analysis of your OMR sheet image
    # Fine-tune these values if needed
    
    # Starting position of the first bubble (top-left)
    start_x = 120  # X position of PYTHON column, first bubble
    start_y = 280  # Y position of question 1
    
    # Spacing between elements
    col_spacing = 160    # Distance between subject columns
    row_spacing = 28     # Distance between question rows
    bubble_spacing = 22  # Distance between A, B, C, D bubbles
    
    # Size of the detection area for all 4 bubbles in a question
    question_width = bubble_spacing * 4  # Width to cover all 4 options
    question_height = 20  # Height of bubble detection area
    
    bubble_roi = {}
    subjects = ["PYTHON", "DATA_ANALYSIS", "MySQL", "POWER_BI", "Adv_STATS"]
    
    for col_idx, subject in enumerate(subjects):
        x_col = start_x + col_idx * col_spacing
        bubble_roi[subject] = []
        
        for row_idx in range(20):  # 20 questions
            y_row = start_y + row_idx * row_spacing
            
            # Store ROI for the entire question (all 4 options)
            # Format: (x, y, width, height, num_options)
            bubble_roi[subject].append((x_col, y_row, question_width, question_height, 4))
    
    return bubble_roi

def generate_rois_calibrated(template_width=1000, template_height=1400):
    """
    Alternative calibrated version with more precise measurements.
    Use this if the above doesn't work perfectly.
    """
    
    # Calibrated positions based on percentage of template size
    # This makes it more robust to different scan resolutions
    
    bubble_roi = {}
    subjects = ["PYTHON", "DATA_ANALYSIS", "MySQL", "POWER_BI", "Adv_STATS"]
    
    # Positions as percentages of template dimensions
    start_x_pct = 0.12  # 12% from left edge
    start_y_pct = 0.20  # 20% from top edge
    
    col_spacing_pct = 0.16   # 16% of width between columns
    row_spacing_pct = 0.020  # 2% of height between rows
    
    bubble_width_pct = 0.088  # Width for all 4 bubbles
    bubble_height_pct = 0.014  # Height of bubble area
    
    for col_idx, subject in enumerate(subjects):
        x_col = int(template_width * (start_x_pct + col_idx * col_spacing_pct))
        bubble_roi[subject] = []
        
        for row_idx in range(20):
            y_row = int(template_height * (start_y_pct + row_idx * row_spacing_pct))
            
            width = int(template_width * bubble_width_pct)
            height = int(template_height * bubble_height_pct)
            
            bubble_roi[subject].append((x_col, y_row, width, height, 4))
    
    return bubble_roi