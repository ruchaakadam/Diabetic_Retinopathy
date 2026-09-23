Yes. Let's stop worrying about Render for now and make the GitHub repository look like a proper polished software/AI project. 🔥
We'll make it portfolio-ready with:

clean folder structure
only the model actually used
no venv, media, outputs, __pycache__, SQLite, .DS_Store
proper README
project screenshots
features + architecture
setup instructions
model information
results
disclaimer
GitHub-friendly presentation
Target GitHub structure
Diabetic_Retinopathy/
│
├── web/
│   ├── manage.py
│   ├── dr_web/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
│   │   └── ...
│   │
│   └── screening/
│       ├── migrations/
│       ├── templates/
│       │   ├── registration/
│       │   └── screening/
│       ├── auth_views.py
│       ├── forms.py
│       ├── models.py
│       ├── urls.py
│       └── views.py
│
├── src/
│   ├── classification/
│   ├── inference/
│   │   ├── predict.py
│   │   └── report_generator.py
│   └── quality/
│
├── models/
│   └── dr_referable_best.pth
│
├── docs/
│   └── screenshots/
│       ├── login.png
│       ├── signup.png
│       ├── upload.png
│       ├── result.png
│       ├── gradcam.png
│       └── report.png
│
├── requirements.txt
├── .gitignore
└── README.md
I would remove these deployment-only files if we're no longer using Render:
build.sh
render.yaml
.python-version
They aren't needed for GitHub itself.
1. First clean the old models
You currently have four models, but your application uses:
dr_referable_best.pth
Keep that one.
Remove the other three from Git only, not from your Mac:

git rm --cached models/dr_efficientnet_b0_balanced_best.pth
git rm --cached models/dr_efficientnet_b0_best.pth
git rm --cached models/dr_efficientnet_b0_finetuned_best.pth
Your local files won't be deleted.
Then your GitHub project will show:

models/
└── dr_referable_best.pth
2. Remove Render-specific files from Git
Since we're focusing on GitHub now:
git rm --cached build.sh
git rm --cached render.yaml
git rm --cached .python-version
Again, this only removes them from Git tracking.
If you want them completely removed locally too:

rm -f build.sh render.yaml .python-version
3. Create a screenshots folder
Run:
mkdir -p docs/screenshots
Now take screenshots of your actual working website.
I'd use these six:

login.png
Your RetinaAI login screen.
signup.png
Your signup page.
upload.png
The retinal image upload screen.
result.png
The main AI result page showing:
image quality
referable probability
decision
recommendation
gradcam.png
The Grad-CAM/explainability section.
report.png
Your beautiful PDF/report viewer with the × button.
Your README can then show the project visually instead of being just text.

4. README structure
Your GitHub README should start with something like:
# RetinaAI 🩺

### Explainable AI for Diabetic Retinopathy Screening

RetinaAI is an AI-assisted diabetic retinopathy screening web application
designed to analyze retinal fundus images, assess image quality, identify
referable diabetic retinopathy, and provide visual explanations using Grad-CAM.

---

## 🚀 Features

- 🔐 User authentication
- 🖼️ Retinal fundus image upload
- 🔍 Automated image-quality assessment
- ✨ Image enhancement for borderline images
- 🤖 EfficientNet-B0 based referable DR classification
- 📊 Referable DR probability
- 🎯 Configurable classification threshold
- 🔥 Grad-CAM explainability
- 📄 Automated screening report generation
- 👤 User-specific screening information
- 📱 Responsive web interface
- ⚕️ Clinical-use disclaimer

---

## 🧠 How RetinaAI Works

```text
Retinal Image
      ↓
Image Quality Assessment
      ↓
 ┌───────────────┐
 │   Good Image  │
 └───────┬───────┘
         ↓
     AI Model
         ↓
Referable Probability
         ↓
   Classification
         ↓
     Grad-CAM
         ↓
 Screening Report
🏗️ Project Structure
Diabetic_Retinopathy/
│
├── web/
│   ├── dr_web/
│   └── screening/
│
├── src/
│   ├── classification/
│   ├── inference/
│   └── quality/
│
├── models/
│   └── dr_referable_best.pth
│
├── docs/
│   └── screenshots/
│
├── requirements.txt
├── .gitignore
└── README.md
🖥️ Screenshots
Login
Image Upload
Screening Result
Grad-CAM Explainability
Screening Report
🤖 AI Model
RetinaAI uses an EfficientNet-B0 based deep-learning classifier
for referable diabetic retinopathy screening.
The screening task is formulated as:

Grade 0–1 → Non-referable
Grade 2–4 → Referable
The current classification threshold is:
0.55
📊 Dataset
The project uses the IDRiD (Indian Diabetic Retinopathy Image Dataset).
The dataset contains retinal fundus images with diabetic retinopathy
severity annotations.

For this project:

Grade 0–1 → Non-referable
Grade 2–4 → Referable
The official test set was kept separate from model training.
🔥 Explainability
RetinaAI uses Grad-CAM to highlight regions of the retinal image
that contributed to the model's prediction.
This provides additional visual context rather than presenting
the prediction as a black-box result.

📄 Generated Report
After screening, RetinaAI generates a structured screening report
containing information such as:
User information
Screening date
Image quality
Classification result
Referable probability
Model information
Explainability information
Recommendation
Medical disclaimer
🛠️ Technology Stack
Frontend
HTML
CSS
JavaScript
Backend
Python
Django
AI / ML
PyTorch
Torchvision
EfficientNet-B0
OpenCV
Scikit-learn
Grad-CAM
Reporting
ReportLab
Database
SQLite for local development
⚙️ Local Setup
Clone the repository:
git clone https://github.com/ruchaakadam/Diabetic_Retinopathy.git
cd Diabetic_Retinopathy
Create a virtual environment:
python3 -m venv venv
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Run migrations:
python web/manage.py migrate
Start the application:
python web/manage.py runserver
Open:
http://127.0.0.1:8000
⚕️ Disclaimer
RetinaAI is an AI-assisted screening prototype developed for
educational and research purposes.
It is not intended to replace examination, diagnosis, or treatment
by a qualified eye-care professional.


---

# 5. One thing I want to improve

Your README should **not claim things we haven't actually implemented**.

For example, we shouldn't write:

> "RetinaAI diagnoses diabetic retinopathy."

Instead:

> "RetinaAI provides AI-assisted screening for referable diabetic retinopathy."

That's much more accurate for your project.

---

# 6. Then push the polished version

After we create the screenshots and README:

```bash
git add .
Check:
git status
Then:
git commit -m "Polish RetinaAI project for GitHub"
and:
git push origin main
One important thing
Your current GitHub repository already has history containing the older models, database, generated PDFs, etc. Cleaning the current structure doesn't erase those files from Git history. For a student portfolio, that's usually fine; the current repository tree will be clean.
If you want the repository to look really professional, the next thing I'd do is make the README visually strong with a RetinaAI banner, screenshots, architecture diagram, model workflow, and results section rather than having it look like a normal college-project repository.
give the whole readme files n tell me where to upload the ss 
Absolutely. Let's make the GitHub README complete and portfolio-ready.
1. Where to put the screenshots
Inside your project, create this exact folder:
diabetic-retinopathy-ai/
│
├── docs/
│   └── screenshots/
│       ├── login.png
│       ├── signup.png
│       ├── upload.png
│       ├── result.png
│       ├── gradcam.png
│       └── report.png
│
├── web/
├── src/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
Create the folder
From your project root:
mkdir -p docs/screenshots
Then put your screenshots inside docs/screenshots/.
For example:

docs/screenshots/login.png
docs/screenshots/signup.png
docs/screenshots/upload.png
docs/screenshots/result.png
docs/screenshots/gradcam.png
docs/screenshots/report.png
You can simply drag the screenshots into that folder using Finder.
Which screenshot should be which?
File	Screenshot
login.png	RetinaAI login page
signup.png	Registration/signup page
upload.png	Image upload/home page
result.png	Screening result page
gradcam.png	Result page showing Grad-CAM
report.png	PDF report viewer with the × button
Don't put the screenshots directly in the root folder. Keep them inside docs/screenshots/.
2. Complete README.md
Your README.md should be directly in the project root:
diabetic-retinopathy-ai/
└── README.md
You can replace the entire contents of your current README.md with the following.
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
🔬 Screening Approach
The project formulates diabetic retinopathy screening as a binary
classification problem.
The original IDRiD diabetic retinopathy grades are mapped as:

DR Grade	Screening Category
Grade 0	Non-referable
Grade 1	Non-referable
Grade 2	Referable
Grade 3	Referable
Grade 4	Referable
Therefore:
Grade 0–1 → Non-referable
Grade 2–4 → Referable
The current screening threshold used by the application is:
0.55
The threshold is applied to the model's predicted referable probability.
🧪 Image Quality Assessment
RetinaAI does not directly classify every uploaded image.
The image first passes through a quality-assessment stage using multiple
image characteristics including:

Focus
Brightness
Contrast
Field of view
A combined quality score is calculated and the image is categorized as:
GOOD
BORDERLINE
UNGRADEABLE
For borderline images, the system can apply image enhancement and perform
the quality assessment again.
Images considered ungradable are not passed directly to the classification
stage.

This helps prevent low-quality retinal images from being interpreted as
normal or abnormal solely because of poor image quality.

🤖 AI Model
RetinaAI uses an EfficientNet-B0 based deep-learning classifier for
referable diabetic retinopathy screening.
The trained model used by the current application is:

models/dr_referable_best.pth
The model was developed using the IDRiD dataset and the project uses a
binary referable/non-referable screening formulation.
📊 Model Evaluation
Evaluation was performed on the official IDRiD test set that was kept
separate from model training.
At the selected threshold of 0.55, the current evaluation produced:

Metric	Value
Accuracy	80.58%
Sensitivity	81.25%
Specificity	79.49%
ROC-AUC	87.50%
False Positives	8
False Negatives	12
These values represent the evaluation performed during development and
should not be interpreted as clinical validation.
🔥 Explainability
RetinaAI uses Grad-CAM to provide visual context for the model's
prediction.
Grad-CAM generates a heatmap showing image regions that contributed to
the model's classification.

This allows the application to provide more information than a simple
classification label.

Example workflow:

Retinal Image
      ↓
AI Prediction
      ↓
Grad-CAM
      ↓
Activation Heatmap
      ↓
Visual Explanation
🖥️ Application Screenshots
🔐 Login
📝 User Registration
🖼️ Retinal Image Upload
📊 Screening Result
🔥 Grad-CAM Explainability
📄 Screening Report
📄 Automated Screening Report
After completing a screening, RetinaAI generates a structured PDF report.
The report can contain:

User name
User email
Screening ID
Screening date
Original image-quality score
Final image-quality score
Quality status
Classification source
Referable probability
Screening decision
Explainability information
Recommendation
Clinical disclaimer
The report can be viewed directly inside the application using the
built-in report viewer.
🏗️ Project Structure
Diabetic_Retinopathy/
│
├── web/
│   ├── manage.py
│   │
│   ├── dr_web/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
│   │   └── ...
│   │
│   └── screening/
│       ├── migrations/
│       │   └── 0001_initial.py
│       │
│       ├── templates/
│       │   ├── registration/
│       │   │   ├── login.html
│       │   │   └── signup.html
│       │   │
│       │   └── screening/
│       │       ├── upload.html
│       │       └── result.html
│       │
│       ├── auth_views.py
│       ├── forms.py
│       ├── models.py
│       ├── urls.py
│       └── views.py
│
├── src/
│   ├── classification/
│   ├── inference/
│   │   ├── predict.py
│   │   └── report_generator.py
│   │
│   └── quality/
│       └── image_quality.py
│
├── models/
│   └── dr_referable_best.pth
│
├── docs/
│   └── screenshots/
│       ├── login.png
│       ├── signup.png
│       ├── upload.png
│       ├── result.png
│       ├── gradcam.png
│       └── report.png
│
├── requirements.txt
├── .gitignore
└── README.md
🛠️ Technology Stack
Frontend
HTML5
CSS3
JavaScript
Backend
Python
Django
Machine Learning
PyTorch
Torchvision
EfficientNet-B0
OpenCV
NumPy
Pandas
Scikit-learn
SciPy
Explainability
Grad-CAM
Report Generation
ReportLab
Database
SQLite for local development
⚙️ Installation
1. Clone the repository
git clone https://github.com/ruchaakadam/Diabetic_Retinopathy.git
Move into the project directory:
cd Diabetic_Retinopathy
2. Create a Virtual Environment
python3 -m venv venv
Activate it on macOS/Linux:
source venv/bin/activate
On Windows:
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run Database Migrations
python web/manage.py migrate
5. Start the Django Server
python web/manage.py runserver
The application will be available at:
http://127.0.0.1:8000
🔑 User Flow
Open RetinaAI
      ↓
Create Account
      ↓
Login
      ↓
Upload Fundus Image
      ↓
Image Quality Assessment
      ↓
Enhancement if Required
      ↓
AI Screening
      ↓
Referable Probability
      ↓
Screening Decision
      ↓
Grad-CAM Explanation
      ↓
Generate Report
📁 Important Files
src/inference/predict.py
Contains the main inference pipeline, including:
Image loading
Quality assessment
Enhancement
Model inference
Threshold-based decision
Grad-CAM generation
Report generation
src/inference/report_generator.py
Responsible for creating the PDF screening report.
src/quality/
Contains the image-quality assessment functionality.
web/screening/views.py
Connects the Django web application with the AI inference pipeline.
web/screening/models.py
Stores screening information associated with authenticated users.
📈 Development Results
The project was developed and evaluated using retinal fundus images from
the IDRiD dataset.
The official IDRiD test set was kept separate from the model-training
images during evaluation.

The selected operating threshold was 0.55 based on the development
threshold analysis.

The application also includes image-quality gating so that ungradable
images can be rejected rather than automatically classified.

🎯 Project Goals
The project focuses on demonstrating how an AI-assisted screening system
can combine:
Computer vision
Deep learning
Explainable AI
Web development
User authentication
Automated reporting
Image-quality assessment
into a single application.
The intended project context is AI-assisted diabetic retinopathy
screening, particularly as a research and educational prototype.

⚕️ Medical Disclaimer
RetinaAI is an educational and research prototype.
The predictions generated by this application are not a medical diagnosis
and should not be used as a substitute for examination, diagnosis, or
treatment by a qualified ophthalmologist or other eye-care professional.

Users should seek appropriate professional medical advice for clinical
decisions.

👩‍💻 Project
RetinaAI — Explainable AI for Diabetic Retinopathy Screening
Developed as a student AI and web-development project.

📜 License
This project is intended for educational and research purposes.
Please review the licensing terms of the IDRiD dataset and any other
third-party resources before redistributing them.


---

## 3. Very important: don't upload the dataset

Your GitHub should **not** contain the full IDRiD dataset.

Don't put:

```text
data/
├── 413 retinal images
├── test images
└── annotations
into the repository.
Your repository only needs your code + trained model + documentation/screenshots.

4. Add the screenshots
After you've taken them, your commands should be:
mkdir -p docs/screenshots
Then your Finder should look like:
docs
└── screenshots
    ├── login.png
    ├── signup.png
    ├── upload.png
    ├── result.png
    ├── gradcam.png
    └── report.png
You can rename your screenshots to those exact names.
5. Create the README from Terminal
If you want to replace your README quickly:
nano README.md
Paste the entire README above.
Then:

Ctrl + O
Enter
Ctrl + X
Check it:
cat README.md
6. Push the polished GitHub version
Once the screenshots are in place:
git add README.md docs/screenshots/
Then:
git status
Make sure you're not accidentally staging:
web/media/
web/outputs/
web/db.sqlite3
venv/
.venv/
__pycache__/
.DS_Store
Then:
git commit -m "Add professional project documentation and screenshots"
and:
git push origin main
Your GitHub repository will then have a much more professional presentation: project → screenshots → architecture → AI methodology → results → setup → disclaimer.
send the whole readme that i can upload on github directly and i have this as my images in the path u mentioned
Yes — and I matched the README to the actual screenshot filenames you showed me.
Download the complete README
Download README.md
Your screenshot folder should be exactly
docs/
└── screenshots/
    ├── gradcam
    ├── login
    ├── report
    ├── result
    └── upload.png
I noticed your screenshots appear to have no .png extension on gradcam, login, report, and result, while upload.png does. The README I created uses those exact names, so GitHub can resolve them as they currently exist.
Put the README here
Diabetic_Retinopathy/
├── README.md              ← replace your current one
├── docs/
│   └── screenshots/
│       ├── gradcam
│       ├── login
│       ├── report
│       ├── result
│       └── upload.png
├── models/
│   └── dr_referable_best.pth
├── src/
└── web/
Then upload to GitHub
After putting the README in your project root:
git add README.md docs/screenshots/
git commit -m "Add professional README and screenshots"
git push origin main
One thing I'd recommend before the final push: rename the four extensionless screenshots to .png so the repository looks cleaner:
gradcam.png
login.png
report.png
result.png
upload.png
If you do that, tell me and I'll give you the updated README with all five .png paths so everything is perfectly clean. 

README.md
Document







README.md


RetinaAI 🩺
Explainable AI for Diabetic Retinopathy Screening
RetinaAI is an AI-assisted web application designed to screen retinal fundus images for referable diabetic retinopathy (DR).
The system combines retinal image-quality assessment, image enhancement, deep-learning classification, Grad-CAM explainability, user authentication, and automated screening-report generation into a single web application.

Medical disclaimer: RetinaAI is an educational and research prototype. It is not intended to replace examination, diagnosis, or treatment by a qualified eye-care professional.
✨ Features
🔐 User registration and login
🖼️ Retinal fundus image upload
🔍 Automated retinal image-quality assessment
✨ Enhancement of borderline-quality images
🤖 EfficientNet-B0 based referable DR classification
📊 Referable DR probability
🎯 Threshold-based screening decision
🔥 Grad-CAM explainability
📄 Automated PDF screening reports
👤 User information in generated reports
📋 Database-backed screening records
🎨 Medical-themed web interface
❌ In-page screening-report viewer with close button
⚕️ Clinical-use disclaimer
🧠 How RetinaAI Works
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
🔬 Screening Approach
The project formulates diabetic retinopathy screening as a binary classification problem.
DR Grade	Screening Category
Grade 0	Non-referable
Grade 1	Non-referable
Grade 2	Referable
Grade 3	Referable
Grade 4	Referable
Grade 0–1 → Non-referable
Grade 2–4 → Referable
The current screening threshold used by the application is:
0.55
🔍 Image Quality Assessment
RetinaAI evaluates the quality of an uploaded retinal image before classification.
The quality pipeline considers:

Focus
Brightness
Contrast
Field of view
Images are categorized as:
GOOD
BORDERLINE
UNGRADEABLE
For borderline images, the system can apply image enhancement and perform quality assessment again.
Images considered ungradable are not directly passed to the classification stage.

🤖 AI Model
RetinaAI uses an EfficientNet-B0 based deep-learning classifier for referable diabetic retinopathy screening.
The primary trained model used by the current application is:

models/dr_referable_best.pth
The project uses the IDRiD (Indian Diabetic Retinopathy Image Dataset) and a binary referable/non-referable screening formulation.
📊 Model Evaluation
Evaluation was performed on the official IDRiD test set kept separate from model training.
At the selected threshold of 0.55, the development evaluation produced:

Metric	Value
Accuracy	80.58%
Sensitivity	81.25%
Specificity	79.49%
ROC-AUC	87.50%
False Positives	8
False Negatives	12
These values represent the project's development evaluation and should not be interpreted as clinical validation.
🔥 Explainable AI with Grad-CAM
RetinaAI uses Grad-CAM to provide visual context for the model's prediction.
Retinal Image
      ↓
AI Prediction
      ↓
Grad-CAM
      ↓
Activation Heatmap
      ↓
Visual Explanation
The explanation provides additional visual context and is not a clinical diagnostic tool.
🖥️ Application Screenshots
All screenshots are stored in:
docs/screenshots/
🔐 Login

🖼️ Retinal Image Upload

📊 Screening Result

🔥 Grad-CAM Explainability

📄 Screening Report

The screenshot filenames above match the files currently in your screenshots folder. If you rename them to include .png, update the corresponding paths in this section.
📄 Automated Screening Report
After a screening is completed, RetinaAI generates a structured PDF report containing information such as:
User name
User email
Screening ID
Screening date
Original image-quality score
Final image-quality score
Quality status
Classification source
Referable probability
Screening decision
Explainability information
Recommendation
Clinical disclaimer
The report can be viewed directly inside the application using the built-in report viewer.
🏗️ Project Structure
Diabetic_Retinopathy/
│
├── web/
│   ├── manage.py
│   ├── dr_web/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
│   │   └── ...
│   │
│   └── screening/
│       ├── migrations/
│       │   └── 0001_initial.py
│       ├── templates/
│       │   ├── registration/
│       │   │   ├── login.html
│       │   │   └── signup.html
│       │   └── screening/
│       │       ├── upload.html
│       │       └── result.html
│       ├── auth_views.py
│       ├── forms.py
│       ├── models.py
│       ├── urls.py
│       └── views.py
│
├── src/
│   ├── classification/
│   ├── inference/
│   │   ├── predict.py
│   │   └── report_generator.py
│   └── quality/
│
├── models/
│   └── dr_referable_best.pth
│
├── docs/
│   └── screenshots/
│       ├── gradcam
│       ├── login
│       ├── report
│       ├── result
│       └── upload.png
│
├── requirements.txt
├── .gitignore
└── README.md
🛠️ Technology Stack
Frontend
HTML5
CSS3
JavaScript
Backend
Python
Django
Machine Learning
PyTorch
Torchvision
EfficientNet-B0
OpenCV
NumPy
Pandas
Scikit-learn
SciPy
Explainability
Grad-CAM
Report Generation
ReportLab
Database
SQLite for local development
⚙️ Local Installation
1. Clone the repository
git clone https://github.com/ruchaakadam/Diabetic_Retinopathy.git
cd Diabetic_Retinopathy
2. Create a virtual environment
macOS/Linux:
python3 -m venv venv
source venv/bin/activate
Windows:
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run database migrations
python web/manage.py migrate
5. Start the application
python web/manage.py runserver
Open:
http://127.0.0.1:8000
🔑 Application Flow
Open RetinaAI
      ↓
Create Account
      ↓
Login
      ↓
Upload Fundus Image
      ↓
Image Quality Assessment
      ↓
Enhancement if Required
      ↓
AI Screening
      ↓
Referable Probability
      ↓
Screening Decision
      ↓
Grad-CAM Explanation
      ↓
Generate Screening Report
📁 Important Files
src/inference/predict.py
Main AI inference pipeline responsible for image loading, quality assessment, enhancement, model inference, threshold-based decision, Grad-CAM generation, and report generation.
src/inference/report_generator.py
Responsible for generating the PDF screening report.
src/quality/
Contains retinal image-quality assessment functionality.
web/screening/views.py
Connects the Django web application with the AI inference pipeline.
web/screening/models.py
Stores screening information associated with authenticated users.
web/screening/result.html
Displays screening results and provides the in-page screening-report viewer.
🎯 Project Goals
RetinaAI demonstrates how multiple technologies can be combined into an AI-assisted screening platform:
Computer vision
Deep learning
Explainable AI
Web development
User authentication
Image-quality assessment
Automated PDF reporting
Database-backed screening records
🔮 Future Improvements
Larger and more diverse retinal datasets
External validation on additional datasets
Improved image-quality assessment
Improved model calibration
Secure cloud-based image storage
Production-grade database infrastructure
Role-based access for healthcare professionals
Detailed screening-history dashboards
Mobile-friendly deployment
Additional retinal disease screening capabilities
⚕️ Medical Disclaimer
RetinaAI is an educational and research prototype.
The predictions generated by this application are not a medical diagnosis and should not be used as a substitute for examination, diagnosis, or treatment by a qualified ophthalmologist or other eye-care professional.

Users should seek appropriate professional medical advice for clinical decisions.

👩‍💻 Project
RetinaAI — Explainable AI for Diabetic Retinopathy Screening
Developed as a student AI, machine-learning, and web-development project.

📜 License
This project is intended for educational and research purposes.
Please review the licensing terms of the IDRiD dataset and other third-party resources before redistributing them.
