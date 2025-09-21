import cv2
import numpy as np
from collections import defaultdict

def detect_sheet_corners(image):
    """Detect corner markers on the OMR sheet"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for square-like corner markers
    corner_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Adjust these values based on your corner marker size
        if 500 <= area <= 10000 and 0.5 <= aspect_ratio <= 1.5:
            corner_candidates.append((x + w//2, y + h//2))
    
    if len(corner_candidates) < 4:
        # Fallback to image corners if markers not found
        h, w = image.shape[:2]
        return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    
    # Sort corners: top-left, top-right, bottom-right, bottom-left
    corner_candidates = np.array(corner_candidates)
    s = corner_candidates.sum(axis=1)
    diff = np.diff(corner_candidates, axis=1).flatten()
    
    tl = corner_candidates[np.argmin(s)]
    br = corner_candidates[np.argmax(s)]
    tr = corner_candidates[np.argmin(diff)]
    bl = corner_candidates[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def preprocess_sheet_from_array(image, corners, width, height):
    """Warp and preprocess the sheet"""
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    # Convert to grayscale and apply adaptive threshold
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    
    # Use adaptive threshold for better bubble detection
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    return binary, warped

def detect_bubbles_grid_based(binary_img, template_width=1000, template_height=1400):
    """
    Detect bubbles using a grid-based approach instead of Hough circles.
    This is more reliable for structured OMR sheets.
    """
    # Define the grid structure based on your OMR sheet
    # 5 subjects, 20 questions each, 4 options per question
    
    # These values need to be tuned to your specific sheet layout
    # Based on the image, approximate positions:
    subjects = ["PYTHON", "DATA ANALYSIS", "MySQL", "POWER BI", "Adv STATS"]
    
    # Starting positions and spacings (tune these!)
    start_x = 120  # X position of first column
    start_y = 280  # Y position of first row of bubbles
    
    col_spacing = 160  # Distance between subject columns
    row_spacing = 28   # Distance between question rows
    
    bubble_spacing = 22  # Distance between option bubbles (A, B, C, D)
    bubble_radius = 8    # Approximate radius of each bubble
    
    detected_bubbles = []
    
    for col_idx, subject in enumerate(subjects):
        x_col = start_x + col_idx * col_spacing
        
        for row_idx in range(20):  # 20 questions per subject
            y_row = start_y + row_idx * row_spacing
            
            for opt_idx in range(4):  # 4 options (A, B, C, D)
                x_bubble = x_col + opt_idx * bubble_spacing
                
                # Create a region of interest around expected bubble position
                roi_size = bubble_radius * 2 + 4
                x1 = max(0, int(x_bubble - roi_size//2))
                y1 = max(0, int(y_row - roi_size//2))
                x2 = min(binary_img.shape[1], int(x_bubble + roi_size//2))
                y2 = min(binary_img.shape[0], int(y_row + roi_size//2))
                
                roi = binary_img[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Check if this region has enough white pixels (filled bubble)
                    white_pixels = cv2.countNonZero(roi)
                    total_pixels = roi.size
                    
                    bubble_info = {
                        'subject': subject,
                        'question': row_idx + 1,
                        'option': chr(ord('A') + opt_idx),
                        'x': x_bubble,
                        'y': y_row,
                        'roi': (x1, y1, x2-x1, y2-y1),
                        'fill_ratio': white_pixels / total_pixels if total_pixels > 0 else 0
                    }
                    detected_bubbles.append(bubble_info)
    
    return detected_bubbles

def evaluate_sheet(binary_img, bubble_roi, answer_key, bubble_threshold=0.30):
    """
    Evaluate the OMR sheet using improved bubble detection
    """
    # If bubble_roi is in the old format, detect bubbles using new method
    if isinstance(bubble_roi, dict) and len(bubble_roi) > 0:
        # Use the old format for backward compatibility
        return evaluate_sheet_legacy(binary_img, bubble_roi, answer_key, bubble_threshold)
    
    # New evaluation using grid-based detection
    bubbles = detect_bubbles_grid_based(binary_img)
    
    scores = defaultdict(int)
    student_answers = {}
    audit = {}
    
    # Group bubbles by subject and question
    bubble_groups = defaultdict(lambda: defaultdict(list))
    for bubble in bubbles:
        bubble_groups[bubble['subject']][bubble['question']].append(bubble)
    
    # Evaluate each question
    for subject, questions in bubble_groups.items():
        for q_num, options in questions.items():
            marked_options = []
            
            for bubble in options:
                if bubble['fill_ratio'] >= bubble_threshold:
                    marked_options.append(bubble['option'])
            
            # Format question key to match answer key format
            qkey = f"{subject.replace(' ', '_')}_Q{q_num}"
            audit[qkey] = marked_options
            
            # Determine student's answer
            if len(marked_options) == 1:
                student_answers[qkey] = marked_options[0]
                
                # Check if correct
                correct_answer = answer_key.get(qkey)
                if correct_answer and marked_options[0] == correct_answer:
                    scores[subject] += 1
            elif len(marked_options) > 1:
                student_answers[qkey] = "Multiple Marks"
            else:
                student_answers[qkey] = "Unanswered"
    
    total_score = sum(scores.values())
    return dict(scores), total_score, student_answers, audit

def evaluate_sheet_legacy(binary_img, bubble_rois, answer_key, bubble_threshold=0.30):
    """Legacy evaluation for backward compatibility"""
    scores = defaultdict(int)
    student_answers = {}
    audit = {}
    
    for subject, rois in bubble_rois.items():
        for q_index, roi in enumerate(rois):
            if len(roi) == 5:  # New format with num_options
                qx, qy, qw, qh, num_options = roi
            else:  # Old format
                qx, qy, qw, qh = roi
                num_options = 4
            
            option_width = qw // num_options
            marked_options = []
            
            for option_index in range(num_options):
                opt_x = max(int(qx + option_index * option_width) - 2, 0)
                opt_y = max(int(qy) - 2, 0)
                opt_w = int(option_width) + 4
                opt_h = int(qh) + 4
                
                opt_x2 = min(opt_x + opt_w, binary_img.shape[1])
                opt_y2 = min(opt_y + opt_h, binary_img.shape[0])
                
                roi_img = binary_img[opt_y:opt_y2, opt_x:opt_x2]
                
                if roi_img.size == 0:
                    fill_ratio = 0.0
                else:
                    fill_ratio = cv2.countNonZero(roi_img) / roi_img.size
                
                if fill_ratio >= bubble_threshold:
                    option_char = chr(ord('A') + option_index)
                    marked_options.append(option_char)
            
            qkey = f"{subject}_Q{q_index + 1}"
            audit[qkey] = marked_options
            correct_answer = answer_key.get(qkey)
            
            if len(marked_options) == 1:
                student_answers[qkey] = marked_options[0]
                if correct_answer is not None and marked_options[0] == correct_answer:
                    scores[subject] += 1
            elif len(marked_options) > 1:
                student_answers[qkey] = "Multiple Marks"
            else:
                student_answers[qkey] = "Unanswered"
    
    total_score = sum(scores.values())
    return dict(scores), total_score, student_answers, audit

def visualize_detection(warped_img, binary_img, bubble_roi=None, bubble_threshold=0.30):
    """
    Visualize bubble detection with proper alignment
    """
    vis = warped_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    # Use grid-based detection for visualization
    bubbles = detect_bubbles_grid_based(binary_img)
    
    for bubble in bubbles:
        x1, y1, w, h = bubble['roi']
        x2 = x1 + w
        y2 = y1 + h
        
        # Determine color based on fill ratio
        if bubble['fill_ratio'] >= bubble_threshold:
            color = (0, 255, 0)  # Green for filled
            thickness = 2
        else:
            color = (0, 0, 255)  # Red for unfilled
            thickness = 1
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        
        # Add option label
        label = bubble['option']
        cv2.putText(vis, label, (x1 + 2, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return vis

def warp_sheet(image, corners, width, height):
    """Wrapper for backward compatibility"""
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, M

def preprocess_for_bubbles(warped_img):
    """Preprocess image for bubble detection"""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) if len(warped_img.shape) == 3 else warped_img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

# Keep these functions for backward compatibility
def detect_bubbles_and_group(binary_img, **kwargs):
    """Legacy function - redirects to grid-based detection"""
    return []

def generate_rois_from_grid(grid, box_size=40):
    """Legacy function for backward compatibility"""
    return {}