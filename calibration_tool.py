"""
Interactive calibration tool to find the exact bubble positions for your OMR sheet.
Run this to get the precise coordinates for your specific template.
"""

import cv2
import numpy as np
import json

class OMRCalibrator:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.display_image = self.image.copy()
        self.points = []
        self.calibration_data = {
            'start_x': None,
            'start_y': None,
            'col_spacing': None,
            'row_spacing': None,
            'bubble_spacing': None
        }
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.display_image, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(self.display_image, f"{len(self.points)}", (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Calibration', self.display_image)
            
    def calibrate(self):
        """
        Interactive calibration process:
        1. Click on first bubble (Python Q1, option A)
        2. Click on second bubble (Python Q1, option B) - for bubble spacing
        3. Click on Python Q2, option A - for row spacing
        4. Click on Data Analysis Q1, option A - for column spacing
        """
        
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibration', self.mouse_callback)
        
        instructions = [
            "Click on PYTHON Q1 Option A (first bubble)",
            "Click on PYTHON Q1 Option B (second bubble)",
            "Click on PYTHON Q2 Option A",
            "Click on DATA ANALYSIS Q1 Option A",
            "Press 'q' when done or 'r' to reset"
        ]
        
        while True:
            display = self.display_image.copy()
            
            # Show instructions
            y_offset = 30
            for i, instruction in enumerate(instructions):
                if i == len(self.points):
                    cv2.putText(display, f">>> {instruction}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif i < len(self.points):
                    cv2.putText(display, f"✓ {instruction}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                else:
                    cv2.putText(display, instruction, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                y_offset += 30
            
            cv2.imshow('Calibration', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.points = []
                self.display_image = self.image.copy()
                
        cv2.destroyAllWindows()
        
        if len(self.points) >= 4:
            self.calculate_calibration()
            return True
        return False
    
    def calculate_calibration(self):
        """Calculate spacing from clicked points"""
        if len(self.points) >= 4:
            # First point is the start position
            self.calibration_data['start_x'] = self.points[0][0]
            self.calibration_data['start_y'] = self.points[0][1]
            
            # Bubble spacing (between A and B)
            self.calibration_data['bubble_spacing'] = self.points[1][0] - self.points[0][0]
            
            # Row spacing (between Q1 and Q2)
            self.calibration_data['row_spacing'] = self.points[2][1] - self.points[0][1]
            
            # Column spacing (between subjects)
            self.calibration_data['col_spacing'] = self.points[3][0] - self.points[0][0]
            
    def save_calibration(self, filename='omr_calibration.json'):
        """Save calibration data to file"""
        with open(filename, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        print(f"Calibration saved to {filename}")
        
    def print_calibration(self):
        """Print calibration values for roi_generator.py"""
        print("\n" + "="*50)
        print("CALIBRATION RESULTS")
        print("="*50)
        print("\nCopy these values to roi_generator.py:\n")
        print(f"start_x = {self.calibration_data['start_x']}")
        print(f"start_y = {self.calibration_data['start_y']}")
        print(f"col_spacing = {self.calibration_data['col_spacing']}")
        print(f"row_spacing = {self.calibration_data['row_spacing']}")
        print(f"bubble_spacing = {self.calibration_data['bubble_spacing']}")
        print("\n" + "="*50)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate OMR bubble positions')
    parser.add_argument('image', help='Path to OMR sheet image')
    parser.add_argument('--save', default='omr_calibration.json', 
                       help='Save calibration to file')
    args = parser.parse_args()
    
    calibrator = OMRCalibrator(args.image)
    
    print("="*50)
    print("OMR CALIBRATION TOOL")
    print("="*50)
    print("\nInstructions:")
    print("1. Click on the center of each bubble as instructed")
    print("2. Press 'q' when done")
    print("3. Press 'r' to reset if you make a mistake")
    print("="*50)
    
    if calibrator.calibrate():
        calibrator.print_calibration()
        calibrator.save_calibration(args.save)
    else:
        print("Calibration incomplete. Need at least 4 points.")

if __name__ == "__main__":
    main()