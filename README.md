# ECG-Vision-AI
A computer vision pipeline that interprets 12-lead ECG images to detect arrhythmias and myocardial infarctions, integrated with GPT-4o for structured clinical reporting.
# 🫀 ECG Vision AI: From Image to Diagnosis

**An end-to-end AI system for 12-lead ECG interpretation.**

This project uses **EfficientNet-B0 (FastAI/PyTorch)** to classify ECG images into 8 clinical categories (including STEMI, AFib, and LBBB) with **99% accuracy**. The prediction is then passed to **OpenAI GPT-4o-mini** to generate a structured, physician-grade clinical report.

---

##  Key Features

* **Vision Model:** Fine-tuned **EfficientNet-B0** trained on the **PTB-XL** clinical dataset (21k+ records).
* **Data Pipeline:** Custom signal-to-image conversion simulating real-world ECG paper grids.
* **Hybrid AI:** Combines Computer Vision (classification) with LLMs (reasoning/reporting).
* **Deployment:** Ready-to-deploy **Streamlit** web application.

##  Tech Stack

* **Frameworks:** FastAI, PyTorch, Streamlit
* **Models:** EfficientNet-B0 (Vision), GPT-4o-mini (LLM)
* **Data:** PTB-XL Dataset (PhysioNet)
* **Tools:** `wfdb` (Signal Processing), `ecg_plot` (Visualization)

##  Performance

| Class | F1-Score | Clinical Significance |
| :--- | :--- | :--- |
| **Normal** | 0.99 | Healthy baseline |
| **AFib** | 0.98 | Atrial Fibrillation (Stroke risk) |
| **Inferior MI** | 1.00 | Heart Attack (Inferior Wall) |
| **Anterior MI** | 0.99 | Heart Attack (Anterior Wall) |
| **LBBB/RBBB** | 0.99 | Bundle Branch Blocks |

*Model achieved an overall **Accuracy of >98%** on the validation set.*

Quick Start
1. Clone the Repository
Bash
git clone [https://github.com/your-username/ECG-Vision-AI.git](https://github.com/your-username/ECG-Vision-AI.git)
cd ECG-Vision-AI
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run the App
Bash
streamlit run app.py
 How It Works
Signal Processing: Raw ECG signals (12-lead) are converted into standardized images with gridlines using ecg_plot.

Vision Inference: The user uploads an image. The EfficientNet-B0 model analyzes the visual patterns (ST-elevation, irregular rhythm, etc.).

LLM Reasoning: The predicted label (e.g., "Inferior MI") and confidence score are sent to GPT-4o.

Clinical Report: The LLM generates a structured JSON report including diagnosis, key findings, and immediate clinical actions.

 Disclaimer
This tool is for educational and research purposes only. It is not a medical device and should not be used for clinical decision-making without physician oversight.

##  Project Structure

```bash
ECG-Vision-AI/
├── app.py                # Streamlit Web Application
├── ecg_model_v1.pkl      # Trained FastAI Model (Export)
├── requirements.txt      # Python Dependencies
├── notebooks/            # Jupyter Notebooks for Training
└── assets/               # Demo images and screenshots
