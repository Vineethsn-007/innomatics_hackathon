"""
Test script to debug bubble detection and visualize results
"""

import cv2
import numpy as np
import json
from omr_processing import preprocess_sheet_from_array, detect_bubbles_grid_based
from roi_generator import generate_rois

def test_bubble_detection(image_path, calibration_file=None):
    """
    Test bubble detection with visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    h, w = image.shape[:2]
    corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    
    # Preprocess
    binary, warped = preprocess_sheet_from_array(image, corners, 1000, 1400)
    
    # Load calibration if provided
    if calibration_file:
        with open(calibration_file, 'r') as f:
            calib = json.load(f)
        print(f"Using calibration from {calibration_file}")
        print(f"Start: ({calib['start_x']}, {calib['start_y']})")
        print(f"Spacings: col={calib['col_spacing']}, row={calib['row_spacing']}, bubble={calib['bubble_spacing']}")
    
    # Detect bubbles
    print("\nDetecting bubbles...")
    bubbles = detect_bubbles_grid_based(binary)
    
    # Create visualization
    vis_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    vis_warped = warped.copy()
    
    # Statistics
    subjects_stats = {}
    filled_count = 0
    
    for bubble in bubbles:
        x1, y1, w, h = bubble['roi']
        x2 = x1 + w
        y2 = y1 + h
        
        # Color based on fill ratio
        if bubble['fill_ratio'] >= 0.30:
            color = (0, 255, 0)  # Green for filled
            filled_count += 1
            thickness = 2
        else:
            color = (0, 0, 255)  # Red for empty
            thickness = 1
        
        # Draw on both images
        cv2.rectangle(vis_binary, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(vis_warped, (x1, y1), (x2, y2), color, thickness)
        
        # Collect statistics
        subject = bubble['subject']
        if subject not in subjects_stats:
            subjects_stats[subject] = {'total': 0, 'filled': 0}
        subjects_stats[subject]['total'] += 1
        if bubble['fill_ratio'] >= 0.30:
            subjects_stats[subject]['filled'] += 1
    
    # Print statistics
    print("\n" + "="*50)
    print("DETECTION STATISTICS")
    print("="*50)
    print(f"Total bubbles detected: {len(bubbles)}")
    print(f"Filled bubbles: {filled_count}")
    print(f"Empty bubbles: {len(bubbles) - filled_count}")
    print("\nBy Subject:")
    for subject, stats in subjects_stats.items():
        print(f"  {subject}: {stats['filled']}/{stats['total']} filled")
    print("="*50)
    
    # Save debug images
    cv2.imwrite("debug_binary_annotated.jpg", vis_binary)
    cv2.imwrite("debug_warped_annotated.jpg", vis_warped)
    print("\nDebug images saved:")
    print("  - debug_binary_annotated.jpg")
    print("  - debug_warped_annotated.jpg")
    
    # Show images (press any key to close)
    cv2.namedWindow("Binary with Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original with Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Binary with Detection", vis_binary)
    cv2.imshow("Original with Detection", vis_warped)
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return bubbles

def analyze_fill_ratios(image_path):
    """
    Analyze fill ratios to determine optimal threshold
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    
    binary, _ = preprocess_sheet_from_array(image, corners, 1000, 1400)
    bubbles = detect_bubbles_grid_based(binary)
    
    # Collect fill ratios
    fill_ratios = [b['fill_ratio'] for b in bubbles]
    fill_ratios.sort()
    
    print("\n" + "="*50)
    print("FILL RATIO ANALYSIS")
    print("="*50)
    print(f"Min fill ratio: {min(fill_ratios):.3f}")
    print(f"Max fill ratio: {max(fill_ratios):.3f}")
    print(f"Mean fill ratio: {np.mean(fill_ratios):.3f}")
    print(f"Median fill ratio: {np.median(fill_ratios):.3f}")
    
    # Show distribution
    print("\nFill Ratio Distribution:")
    bins = np.arange(0, 1.1, 0.1)
    hist, _ = np.histogram(fill_ratios, bins=bins)
    for i, count in enumerate(hist):
        bar = '█' * int(count / 2)
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} ({count})")
    
    # Suggest threshold
    # Look for gap between filled and unfilled
    sorted_ratios = np.array(fill_ratios)
    gaps = np.diff(sorted_ratios)
    max_gap_idx = np.argmax(gaps)
    suggested_threshold = (sorted_ratios[max_gap_idx] + sorted_ratios[max_gap_idx + 1]) / 2
    
    print(f"\nSuggested threshold: {suggested_threshold:.3f}")
    print("(Based on largest gap in fill ratios)")
    print("="*50)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test OMR bubble detection')
    parser.add_argument('image', help='Path to OMR sheet image')
    parser.add_argument('--calibration', help='Path to calibration JSON file')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze fill ratios to find optimal threshold')
    args = parser.parse_args()
    
    if args.analyze:
        analyze_fill_ratios(args.image)
    else:
        test_bubble_detection(args.image, args.calibration)

if __name__ == "__main__":
    main()