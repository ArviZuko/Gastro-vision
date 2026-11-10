# AI-Based Clinical Diagnostic System - Cancer Detection

An advanced multi-architecture AI system for real-time detection and screening of cancer pathologies.

The application, installer, model weights, demonstration video, and dataset images are stored directly in their repository folders using **Git LFS**. Install Git LFS before cloning, or run `git lfs pull` in an existing clone to retrieve the complete files.

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

1. **Clone the repository with Git LFS**

   ```bash
   git lfs install
   git clone https://github.com/ArviZuko/Gastro-vision.git
   cd Gastro-vision
   git lfs pull
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
