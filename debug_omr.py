#!/usr/bin/env python3
"""
Enhanced debugging script for OMR system
"""
import cv2
import numpy as np
from omr_contour_mode import evaluate_contour_mode, SubjectNames
from load_answer_key import load_answer_key_from_sheet

def debug_omr_detection(image_path, answer_key_path, sheet_name="Set - A"):
    """
    Comprehensive debugging of OMR detection and grading
    """
    print("🔍 OMR DEBUG ANALYSIS")
    print("="*50)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📷 Image loaded: {img.shape}")
    
    # Process with different thresholds to find optimal
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    for thresh in thresholds:
        print(f"\n🎯 Testing threshold: {thresh}")
        answers, _, warped, binary = evaluate_contour_mode(
            img, threshold=thresh, margin=0.05
        )
        
        # Count detected answers
        total_detected = 0
        for subject in SubjectNames:
            subject_count = len([q for q in range(1, 21) if answers.get(subject, {}).get(q, "")])
            total_detected += subject_count
            print(f"  {subject}: {subject_count}/20 answers detected")
        
        print(f"  Total: {total_detected}/100 answers detected")
        
        # Save debug image for this threshold
        debug_filename = f"debug_thresh_{thresh:.2f}.jpg"
        cv2.imwrite(debug_filename, warped)
        print(f"  💾 Saved debug image: {debug_filename}")
    
    # Load answer key and analyze format
    print(f"\n📋 Loading answer key from: {answer_key_path}")
    try:
        key = load_answer_key_from_sheet(answer_key_path, sheet_name)
        print(f"✅ Loaded {len(key)} answer key entries")
        
        # Analyze answer key format
        print("\n🔍 Answer key analysis:")
        subjects_in_key = set()
        for key_name in key.keys():
            if " Q" in key_name:
                subject = key_name.split(" Q")[0]
                subjects_in_key.add(subject)
        
        print(f"Subjects in key: {subjects_in_key}")
        print(f"Expected subjects: {set(SubjectNames)}")
        
        # Show sample entries
        print("\nSample answer key entries:")
        for i, (k, v) in enumerate(list(key.items())[:10]):
            print(f"  {k}: '{v}'")
        
    except Exception as e:
        print(f"❌ Error loading answer key: {e}")
        return
    
    # Final analysis with recommended threshold
    recommended_thresh = 0.16
    print(f"\n🎯 Final analysis with recommended threshold: {recommended_thresh}")
    
    answers, _, warped, binary = evaluate_contour_mode(
        img, threshold=recommended_thresh, margin=0.05
    )
    
    # Detailed subject analysis
    print("\n📊 Detailed Detection Analysis:")
    for subject in SubjectNames:
        print(f"\n{subject}:")
        for q in range(1, 21):
            detected = answers.get(subject, {}).get(q, "")
            key_lookup = f"{subject} Q{q}"
            expected = key.get(key_lookup, "NOT_FOUND")
            
            status = "✅" if detected == expected else "❌" if detected else "⚠️"
            print(f"  Q{q:2d}: '{detected:1s}' (expected: '{expected}') {status}")
    
    print("\n✨ Debug analysis complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python debug_omr.py <image_path> <answer_key_path> [sheet_name]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    answer_key_path = sys.argv[2]
    sheet_name = sys.argv[3] if len(sys.argv) > 3 else "Set - A"
    
    debug_omr_detection(image_path, answer_key_path, sheet_name)
