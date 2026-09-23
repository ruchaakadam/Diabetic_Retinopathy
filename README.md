# RetinaAI 🩺

## Explainable AI for Diabetic Retinopathy Screening

RetinaAI is an AI-assisted web application for screening retinal fundus
images for **referable diabetic retinopathy (DR)**.

The system combines image-quality assessment, deep-learning based
classification, image enhancement, Grad-CAM explainability, and automated
screening-report generation into a single web application.

> **Note:** RetinaAI is an educational/research prototype and is not
> intended to replace examination, diagnosis, or treatment by a qualified
> eye-care professional.

---

# ✨ Features

- 🔐 User registration and login
- 🖼️ Retinal fundus image upload
- 🔍 Automated retinal image-quality assessment
- ✨ Image enhancement for borderline-quality images
- 🤖 EfficientNet-B0 based referable DR classification
- 📊 Referable DR probability
- 🎯 Threshold-based screening decision
- 🔥 Grad-CAM explainability
- 📄 Automated PDF screening reports
- 👤 User information in generated reports
- 📋 Screening history stored for authenticated users
- 🎨 Clean medical-themed web interface
- ❌ In-page PDF report viewer with close button
- ⚕️ Clinical-use disclaimer

---

# 🧠 How RetinaAI Works

```text
                 Retinal Fundus Image
                         │
                         ▼
                Image Quality Check
                         │
              ┌──────────┴──────────┐
              │                     │
          Good Quality          Borderline
              │                     │
              │              Image Enhancement
              │                     │
              └──────────┬──────────┘
                         ▼
                  Quality Re-check
                         │
                         ▼
                EfficientNet-B0
                         │
                         ▼
             Referable DR Probability
                         │
                         ▼
                Threshold Decision
                         │
              ┌──────────┴──────────┐
              │                     │
       Non-Referable             Referable
              │                     │
              └──────────┬──────────┘
                         ▼
                    Grad-CAM
                         │
                         ▼
                Screening Report
