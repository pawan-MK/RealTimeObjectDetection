


📝 README File Structure

 1. Project Title

* `# Real-Time Sign Language Detector Project`

 2. Description:
* "This project is a real-time sign language detector built using Python, TensorFlow, and OpenCV. It is a five-member team project for the CAPSTONE.
3. Team Members & Roles
* **Member 1 (Setup Lead):** [Goutham] - Environment and API setup.
* **Member 2 (Data Collector):** [Rohini] - Image and dataset collection.
* **Member 3 (Data Prep):** [ Vamsi] - Data labeling (XML) and TFRecord conversion.
* **Member 4 (Training Lead):** [Anusha ] - Model configuration and training (Azure/local).
* **Member 5 (Deployment Lead):** [Mounika] - Final application scripting and testing.

 4. Technology Used:
A bulleted list of the main technologies.
* Python
* Conda
* TensorFlow Object Detection API
* OpenCV
* Pandas
* Jupyter Notebook 
* Azure (for training/deployment)

5. Setup & Installation

*  Clone the repository:
    * `git clone https://github.com/pawan-MK/RealTimeObjectDetection.git`
    * `cd RealTimeObjectDetection`

*  Create the Conda Environment:
    * `conda create -n sign_env python=3.9`
    * `conda activate sign_env`

*  Install Dependencies:
    * `pip install tensorflow opencv-python matplotlib jupyter pandas`
    * `pip install tensorflow-object-detection-api` 

*  Install Protobufs: 
    * `cd models/research`
    * `protoc object_detection/protos/*.proto --python_out=.`
    * `pip install .`

 6. How to Run the Project
* To run the live detection (Phase 4):
    * `python 3_live_detect.py`

---
Just copy, paste, and edit this structure into a new file named `README.md` in your main project folder. Once you push this file, it will automatically show up on your GitHub project's main page.
