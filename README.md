# AI-Based Clinical Diagnostic System - Cancer Detection

An advanced multi-architecture AI system for real-time detection and screening of cancer pathologies.

Clone the repository for the tracked project files. The [Windows installer](SRC/Applications/Installer/Installer.md), [demo video](DOCUMENTATION/video.md), and [screen recording](Screenshots/video.md) are available from GitHub Releases. The packaged application folders also contain download notes for the [Polars runtime](SRC/Applications/App/_internal/_polars_runtime_32/Libraries.md) and [PyTorch library](SRC/Applications/App/_internal/torch/lib/Libraries.md). See the [dataset guide](Dataset/README.md) for the Kaggle download.

## Repository Structure

`
project-root/
+-- Dataset/
¦   +-- Report/
+-- Models/
+-- App/
¦   +-- .exe/
¦   +-- ISS/
¦   +-- Source Code/
+-- Document/
¦   +-- Comprehensive_Doc/
¦   +-- PPT/
+-- Evaluation/
+-- Output/
+-- .gitignore
+-- README.md
+-- requirements.txt
`

## Quick Start & Setup Guide

1. **Clone the repository**

   ```bash
   git clone https://github.com/ArviZuko/Gastro-vision.git
   cd Gastro-vision
   ```

2. **Set Up Virtual Environment (Recommended)**
   `ash
   python -m venv venv
   venv\Scripts\activate # On Windows
   # source venv/bin/activate # On Linux/macOS
   `

3. **Install Dependencies**
   `ash
   pip install -r requirements.txt
   `

4. **Launch Main Application**
   `ash
   python "App/Source Code/main.py"
   `

## License & Citation
Licensed under the MIT License. Developed by SMARTLAB-SU.
