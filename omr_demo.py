import cv2
import numpy as np

# Load image
image = cv2.imread("omr_sample.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess: blur + threshold
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blurred, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours (possible bubbles)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

bubbles = []
for c in contours:
    (x,y,w,h) = cv2.boundingRect(c)
    # Filter by size (tweak depending on your bubble size)
    if 15 < w < 60 and 15 < h < 60:
        bubbles.append((x,y,w,h))

# Sort bubbles top-to-bottom, left-to-right
bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

# Example: 5 questions × 4 options
answer_key = {0:1, 1:3, 2:0, 3:2, 4:1}  # Q:Correct Option index

score = 0
student_answers = {}

num_questions = len(answer_key)

for (q, i) in enumerate(range(0, num_questions * 4, 4)):
    options = bubbles[i:i+4]  # 4 options per question
    filled = -1
    for (j,(x,y,w,h)) in enumerate(options):
        roi = thresh[y:y+h, x:x+w]
        total = cv2.countNonZero(roi)
        if total > 200:  # threshold for filled bubble
            filled = j
    student_answers[f"Q{q+1}"] = filled
    if filled == answer_key[q]:
        score += 1

print("\n--- OMR Evaluation Results ---")
print(f"Final Score: {score}/{len(answer_key)}")
print("Student Answers:", student_answers)
