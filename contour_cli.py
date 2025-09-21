import cv2
import argparse
from omr_contour_mode import evaluate_contour_mode, SubjectNames
from load_answer_key import load_answer_key_from_sheet

def main():
    p = argparse.ArgumentParser(description="Grid-free OMR grading (5 subjects, 100 questions)")
    p.add_argument("--image", required=True, help="Path to OMR sheet image")
    p.add_argument("--key", required=True, help="Path to answer key Excel")
    p.add_argument("--set", default="Set - A", help="Sheet name in Excel")
    p.add_argument("--width", type=int, default=1000, help="Warped width")
    p.add_argument("--height", type=int, default=1400, help="Warped height")
    p.add_argument("--threshold", type=float, default=0.16, help="Absolute fill floor (0..1)")
    p.add_argument("--outvis", default="debug_contour_vis.jpg", help="Output visualization path")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    # Detect answers for all subjects and 20 rows each
    answers, _, warped, _ = evaluate_contour_mode(
        img, args.width, args.height, threshold=args.threshold, margin=0.05
    )
    
    # Load and normalize answer key
    key = load_answer_key_from_sheet(args.key, args.set)
    print(f"✅ Loaded {len(key)} answers from {args.set}")

    # Debug: Print detected answers
    print("\n🔍 DEBUG: Detected answers:")
    for subject in SubjectNames:
        print(f"\n{subject}:")
        for q in range(1, 21):
            detected = answers.get(subject, {}).get(q, "")
            print(f"  Q{q}: '{detected}'")

    # Debug: Print answer key format
    print("\n🔍 DEBUG: Answer key format:")
    for key_name, answer in list(key.items())[:10]:  # Show first 10
        print(f"  {key_name}: '{answer}'")

    # Grade by subject and total
    total = 0
    correct = 0
    per_subject = {s: {"correct":0, "total":0} for s in SubjectNames}

    print("\n📊 GRADING RESULTS:")
    print("="*60)

    for s in SubjectNames:
        print(f"\n{s}:")
        for q in range(1, 21):
            expected = ""
            possible_keys = [
                f"{s} Q{q}",
                f"{s}Q{q}",
                f"{s}_Q{q}",
                f"{s.replace(' ', '_')} Q{q}",
                f"{s.replace(' ', '')} Q{q}"
            ]
            
            for k in possible_keys:
                if k in key:
                    expected = key[k]
                    break
            
            got = answers.get(s, {}).get(q, "")
            
            per_subject[s]["total"] += 1
            total += 1
            
            ok = False
            if got and expected:
                # Normalize both answers for comparison
                expected_normalized = expected.upper().strip()
                got_normalized = got.upper().strip()
                
                # Handle multiple answers (comma-separated)
                if ',' in expected_normalized:
                    expected_set = set(expected_normalized.split(','))
                    got_set = set(got_normalized.split(',')) if ',' in got_normalized else {got_normalized}
                    ok = expected_set == got_set
                else:
                    ok = expected_normalized == got_normalized
            elif not got and not expected:
                ok = True  # Both empty is considered correct
            
            if ok:
                per_subject[s]["correct"] += 1
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            # Print detailed comparison for debugging
            print(f"  Q{q:2d}: Got='{got}' | Expected='{expected}' | {status}")

    # Report final scores
    print("\n" + "="*60)
    print(f"📊 FINAL SCORE: {correct}/{total} ({correct/total*100:.1f}%)")
    print("="*60)
    
    for s in SubjectNames:
        cs = per_subject[s]["correct"]
        ts = per_subject[s]["total"]
        percentage = (cs/ts*100) if ts > 0 else 0
        print(f"{s:15}: {cs:2d}/{ts:2d} ({percentage:5.1f}%)")

    # Enhanced visualization with answers and correctness
    vis = warped.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_offsets = [60, 260, 460, 660, 860]  # rough left edges of each subject column
    y0, dy = 190, 28                      # top row y and row spacing

    for si, s in enumerate(SubjectNames):
        x0 = x_offsets[si] if si < len(x_offsets) else 60 + 200*si
        for q in range(1, 21):
            y = y0 + (q-1)*dy
            
            # Get detected answer and expected answer
            detected = answers.get(s, {}).get(q, "")
            expected = ""
            for k in [f"{s} Q{q}", f"{s}Q{q}", f"{s}_Q{q}"]:
                if k in key:
                    expected = key[k]
                    break
            
            # Determine if correct
            is_correct = False
            if detected and expected:
                if ',' in expected:
                    is_correct = set(expected.upper().split(',')) == set(detected.upper().split(','))
                else:
                    is_correct = expected.upper() == detected.upper()
            elif not detected and not expected:
                is_correct = True
            
            # Choose color based on correctness
            color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, red if wrong
            if not detected:
                color = (0, 165, 255)  # Orange if no answer detected
            
            # Display detected answer
            display_text = detected if detected else "?"
            cv2.putText(vis, display_text, (x0, y), font, 0.55, color, 1, cv2.LINE_AA)

    cv2.imwrite(args.outvis, vis)
    print(f"\n💾 Saved enhanced visualization to {args.outvis}")

if __name__ == "__main__":
    main()
