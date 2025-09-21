# 📄 Automated OMR Grading System

An advanced Optical Mark Recognition (OMR) system designed for automated grading of multiple-choice answer sheets. This system uses computer vision and machine learning techniques to detect filled bubbles and grade them against answer keys.

## 🌟 Features

- **Grid-free contour detection** - Works without precise alignment
- **5 subjects × 20 questions = 100 total questions** support
- **Robust bubble detection** using contours and Hough circles
- **Streamlit web interface** for easy use
- **Command-line interface** for batch processing
- **Excel answer key support** with flexible formatting
- **Visual debugging** with overlays showing detected answers
- **Detailed grading reports** with subject-wise breakdown
- **Export results** to Excel format

## 📋 Supported OMR Sheet Format

The system is designed for OMR sheets with the following structure:
- **5 columns** representing different subjects:
  - PYTHON (Questions 1-20)
  - DATA ANALYSIS (Questions 21-40)
  - MySQL (Questions 41-60)
  - POWER BI (Questions 61-80)
  - Adv STATS (Questions 81-100)
- **20 rows** per subject
- **4 options** (A, B, C, D) per question

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- streamlit
- opencv-python
- numpy
- pandas
- Pillow
- openpyxl
- matplotlib

### Web Interface (Recommended)

1. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload files:**
   - Upload your OMR sheet image (JPG/PNG)
   - Upload your answer key Excel file
   - Select the answer key sheet (Set-A or Set-B)

4. **Adjust parameters** if needed:
   - Bubble Fill Threshold (0.16 default)
   - Selection Margin (0.05 default)
   - Image dimensions

5. **View results:**
   - Real-time processing visualization
   - Subject-wise score breakdown
   - Detailed answer analysis
   - Download results as Excel

### Command Line Interface

```bash
python contour_cli.py --image omr_sample.jpg --key answer_key.xlsx --set "Set - A" --threshold 0.16
```

**Parameters:**
- `--image`: Path to OMR sheet image
- `--key`: Path to answer key Excel file
- `--set`: Sheet name in Excel ("Set - A" or "Set - B")
- `--width`: Processed image width (default: 1000)
- `--height`: Processed image height (default: 1400)
- `--threshold`: Bubble fill threshold (0.0-1.0, default: 0.16)
- `--outvis`: Output visualization file (default: debug_contour_vis.jpg)

## 📁 Project Structure

```
├── app.py                      # Streamlit web application
├── contour_cli.py             # Command-line interface
├── omr_contour_mode.py        # Core OMR processing engine
├── load_answer_key.py         # Excel answer key loader
├── debug_load_answer_key.py   # Debug utility for answer keys
├── requirements.txt           # Python dependencies
├── answer_key.xlsx           # Sample answer key file
├── omr_sample.jpg           # Sample OMR sheet
└── README.md                # This file
```

### Core Components

1. **app.py** - Full-featured web interface with:
   - File upload handling
   - Real-time parameter adjustment
   - Visual results display
   - Excel export functionality

2. **omr_contour_mode.py** - Advanced OMR processing with:
   - Contour-based bubble detection
   - Robust grid clustering
   - Conservative answer selection
   - Visualization generation

3. **load_answer_key.py** - Answer key processing:
   - Excel file parsing
   - Subject name mapping
   - Multiple answer format support

4. **contour_cli.py** - Command-line tool for:
   - Batch processing
   - Automated workflows
   - Debug output generation

## 📊 Answer Key Format

Your Excel file should have the following structure:

### Sheet: "Set - A" or "Set - B"
```
| Python  | EDA     | SQL     | POWER BI | Statistics |
|---------|---------|---------|----------|------------|
| 1 - a   | 21 - a  | 41 - c  | 61 - b   | 81. a      |
| 2 - c   | 22 - d  | 42 - c  | 62 - c   | 82. b      |
| ...     | ...     | ...     | ...      | ...        |
```

**Supported formats:**
- `1 - a` (dash format)
- `81. a` (dot format)  
- `21 : b` (colon format)
- `16 - a,b,c,d` (multiple answers)

**Column headers mapping:**
- `Python` → PYTHON
- `EDA` → DATA ANALYSIS
- `SQL` → MySQL
- `POWER BI` → POWER BI
- `Statistics`/`Satistics` → Adv STATS

## ⚙️ Configuration

### Processing Parameters

- **Bubble Fill Threshold** (0.0-1.0): Minimum fill ratio to consider a bubble as marked
  - Lower = more sensitive (detects lighter marks)
  - Higher = less sensitive (requires darker marks)
  - Default: 0.16

- **Selection Margin** (0.0-0.2): Required difference between best and second-best bubble
  - Prevents ambiguous selections
  - Default: 0.05

- **Image Dimensions**: Resize parameters for consistent processing
  - Width: 1000px (default)
  - Height: 1400px (default)

### Subject Configuration

Edit `SubjectNames` in `omr_contour_mode.py` to match your subjects:

```python
SubjectNames = ["PYTHON", "DATA ANALYSIS", "MySQL", "POWER BI", "Adv STATS"]
```

## 🔧 Advanced Usage

### Custom Answer Key Mapping

Modify `SUBJECT_MAP` in `load_answer_key.py` for different header names:

```python
SUBJECT_MAP = {
    "PYTHON": "PYTHON",
    "EDA": "DATA ANALYSIS",
    "SQL": "MySQL",
    "POWER BI": "POWER BI",
    "STATISTICS": "Adv STATS",
    "SATISTICS": "Adv STATS"  # Handle typos
}
```

### Debug Mode

Use the debug utility to troubleshoot answer key loading:

```bash
python -c "from debug_load_answer_key import load_answer_key_from_sheet_debug; load_answer_key_from_sheet_debug('answer_key.xlsx', 'Set - A')"
```

### Batch Processing

Process multiple OMR sheets:

```bash
for file in *.jpg; do
    python contour_cli.py --image "$file" --key answer_key.xlsx --set "Set - A"
done
```

## 📈 Output and Results

### Console Output
- Detection summary for each subject
- Question-by-question comparison
- Final score and percentage
- Subject-wise breakdown

### Visualization
- `debug_contour_vis.jpg`: Processed image with detected answers
- Color coding: Green (correct), Red (wrong), Orange (not detected)

### Excel Export (Web Interface)
- Summary sheet with overall metrics
- Detailed results with question-by-question analysis
- Subject-wise performance breakdown

## 🐛 Troubleshooting

### Common Issues

1. **Low detection accuracy:**
   - Adjust bubble fill threshold
   - Check image quality and lighting
   - Verify OMR sheet alignment

2. **Wrong answer key matching:**
   - Check Excel column headers
   - Verify question numbering format
   - Use debug mode to inspect parsing

3. **Import errors:**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility (3.7+)

4. **Memory issues with large images:**
   - Reduce image dimensions
   - Process in smaller batches

### Performance Tips

- Use images with good contrast
- Ensure proper lighting when scanning
- Keep OMR sheets flat and aligned
- Use consistent bubble filling (completely filled)

## 🔬 Technical Details

### Algorithm Overview

1. **Preprocessing**: Resize, grayscale conversion, bilateral filtering
2. **Thresholding**: Adaptive threshold with dynamic parameters
3. **Contour Detection**: Find circular/elliptical shapes
4. **Candidate Filtering**: Size, aspect ratio, and circularity checks
5. **Grid Clustering**: Quantile-based column detection, linear row spacing
6. **Answer Selection**: Conservative voting with margin requirements
7. **Grading**: Comparison with answer key using normalized matching

### Key Technologies

- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and Excel I/O
- **Streamlit**: Web interface framework
- **Matplotlib**: Visualization and plotting

## 📄 License

This project is provided as-is for educational and commercial use. Please ensure compliance with your institution's policies regarding automated grading systems.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample OMR sheets
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Use debug mode to diagnose problems
3. Verify your answer key format
4. Test with provided sample files

---

**Happy Grading! 🎯**
