import cv2
import numpy as np
import argparse
from omr_processing import (detect_sheet_corners, warp_sheet, detect_bubbles_and_group,
                            generate_rois_from_grid, evaluate_sheet, visualize_detection)
from load_answer_key import load_answer_key_from_sheet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to student OMR image")
    parser.add_argument("--key", required=True, help="Path to answer key Excel")
    parser.add_argument("--set", default="Set - A", help="Sheet name in Excel")
    parser.add_argument("--out", default="student_result.xlsx", help="Output Excel file")
    parser.add_argument("--width", type=int, default=1000, help="Warp width")
    parser.add_argument("--height", type=int, default=1400, help="Warp height")
    parser.add_argument("--threshold", type=float, default=0.35, help="Bubble fill threshold")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    corners = detect_sheet_corners(img)
    warped, _ = warp_sheet(img, corners, args.width, args.height)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    bubble_grid = detect_bubbles_and_group(binary)
    bubble_roi = generate_rois_from_grid(bubble_grid)

    answer_key = load_answer_key_from_sheet(args.key, args.set)

    scores, total_score, student_answers, audit = evaluate_sheet(binary, bubble_roi, answer_key, bubble_threshold=args.threshold)
    
    vis = visualize_detection(warped, binary, bubble_roi, bubble_threshold=args.threshold)
    cv2.imwrite("student_result_vis.jpg", vis)

    # Save results - your existing code to save Excel output etc goes here
    print("Marks by Subject:")
    for subject, score in scores.items():
        print(f"{subject}: {score} / 20")  # Assuming 20 questions per subject

    print(f"\nTotal Score: {total_score} / 100")  # Assuming 5 subjects total

    print("\nDetailed Answers:")
    for qkey, ans in student_answers.items():
        print(f"{qkey}: {ans}")

if __name__ == "__main__":
    main()
