# 🧠 Brain Tumor Detection System

A classical computer vision pipeline for detecting and classifying brain tumors from MRI scans — built for an Image Processing course project.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 About

This project implements a full image processing pipeline to detect brain tumors in MRI scans using **classical computer vision techniques** — no deep learning. It classifies tumors into four types (Glioma, Meningioma, Pituitary, Metastatic) and provides a clean web interface for uploading scans and viewing annotated results.

> ⚠️ **Disclaimer:** This is an academic project. Results are not clinically validated. Do not use for real medical diagnosis.

---

## 🔬 Detection Pipeline

```
Input MRI
    │
    ▼
1. Preprocessing
   ├── Grayscale conversion
   ├── Bilateral filter (edge-preserving denoise)
   └── CLAHE contrast enhancement
    │
    ▼
2. Skull Stripping
   ├── Otsu's thresholding
   ├── Morphological closing (fill holes)
   └── Largest contour = brain region
    │
    ▼
3. Tumor Segmentation
   ├── Adaptive intensity threshold (mean + 1.5σ)
   ├── Morphological open/close (noise removal)
   └── Contour detection & filtering
    │
    ▼
4. Feature Extraction
   ├── Circularity  (4π·Area / Perimeter²)
   ├── Area ratio
   ├── Aspect ratio & extent
   ├── Mean & std intensity
   └── Distance from center
    │
    ▼
5. Rule-Based Classification
   ├── Glioma      → irregular, large, interior, heterogeneous
   ├── Meningioma  → circular, peripheral, homogeneous
   ├── Pituitary   → small, sella turcica region, circular
   └── Metastatic  → small, high contrast, irregular
    │
    ▼
6. Annotated Output
   ├── Bounding box + corner markers
   ├── Contour overlay
   └── Tumor type label
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Then open your browser at **http://localhost:5000**

---

## 📁 Project Structure

```
brain_tumor_detection/
├── app.py                  # Flask web server
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend UI
├── utils/
│   └── detector.py         # Core CV pipeline
└── README.md
```

---

## 🧪 Image Processing Techniques Used

| Technique | Purpose |
|-----------|---------|
| **Bilateral Filter** | Edge-preserving noise reduction |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization |
| **Otsu's Thresholding** | Optimal global binarization |
| **Morphological Operations** | Skull stripping, noise removal, hole filling |
| **Contour Detection** | Region boundary finding |
| **Feature Extraction** | Circularity, area, intensity statistics |
| **Rule-based Scoring** | Tumor type classification |

---

## 🔍 Tumor Type Classification

| Type | Key Visual Features |
|------|-------------------|
| **Glioma** | Irregular shape, large area, interior location, heterogeneous signal |
| **Meningioma** | Well-defined border, peripheral location, homogeneous signal |
| **Pituitary** | Small size, located in sellar region, circular shape |
| **Metastatic** | Small, high contrast, can be multiple lesions |

---

## 📊 Dataset

For testing, use the publicly available:
- [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,023 images)
- [Brain Tumor Classification — Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

These datasets include labeled Glioma, Meningioma, Pituitary, and Normal MRI scans.

---

## 🖥️ Screenshots

> Upload an MRI → Get annotated result with tumor type, location, grade, and confidence.

---

## 📚 References

1. Otsu, N. (1979). *A threshold selection method from gray-level histograms.* IEEE Transactions on Systems, Man, and Cybernetics.
2. Reza, A. M. (2004). *Realization of the Contrast Limited Adaptive Histogram Equalization (CLAHE).*
3. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing (4th ed.)*
4. Havaei, M. et al. (2017). *Brain tumor segmentation with deep neural networks.* Medical Image Analysis.

---

## 🛠️ Future Enhancements

- [ ] Deep learning model (CNN / U-Net) trained on labeled dataset
- [ ] Pixel-level segmentation mask instead of bounding box
- [ ] DICOM file format support
- [ ] Grad-CAM visualization
- [ ] Multi-slice 3D analysis
- [ ] Export report as PDF

---

## 👨‍💻 Author

Developed for **Image Processing** course project.

---

## 📄 License

MIT License — free to use for educational purposes.
