Wildfire Detection System 🔥
A web application for detecting wildfires in satellite imagery using deep learning models.

📁 Project Structure

Wildfire-Detection-App/
├── static/
│   ├── css/
│   │   └── 404.html
│   ├── js/
│   │   └── app.js
│   ├── models/
│   └── uploads/
├── templates/
│    ├── index.html
│    └── 404.html
├── wildfire_env/
├── app.py
├── requirements.txt
└── README.md # This file

📸 Interface Overview

Web Application Interface
Main Interface:
![image](https://github.com/user-attachments/assets/30df6c73-f705-425e-8661-6323ae910a98)

Wildfire Detection Result
![1](https://github.com/user-attachments/assets/0771883d-a2da-4be4-9960-5509f2f57222)

Analysis Report
![image](https://github.com/user-attachments/assets/addc88fd-20bd-4822-8229-fd39944a42ad)

✨ Features

- Satellite image analysis (JPEG, PNG, TIFF)
- Custom model upload support (.h5, .keras)
- Real-time wildfire probability prediction
- Interactive heatmap visualization
- Confidence-based results display
- Text report generation

🚀 How to Run

1. Clone repository
cd wildfire-detection

2. Set up virtual environment**
python -m venv wildfire_env
source wildfire_env/bin/activate  # Linux/Mac
wildfire_env\Scripts\activate    # Windows


3. Install dependencies
pip install -r requirements.txt


4. Create required directories

mkdir -p static/uploads static/models


💻 Usage

1. Start the application
flask run --host=0.0.0.0 --port=5000

2. Access the interface at
   http://localhost:5000

4. **Workflow**:
   - Upload satellite image (JPEG/PNG/TIFF)
   - Load trained model (.h5/.keras)
   - Click "Analyze" for detection
   - View visual results & confidence percentage
   - Download text report



Note: Ensure your requirements.txt contains:
flask==3.0.2
tensorflow==2.15.0
numpy==1.24.4
pillow==10.2.0
matplotlib==3.8.3
tifffile==2024.2.12


Adjust versions according to your actual dependencies using:

pip freeze > requirements.txt
