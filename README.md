🤖 Robot Telemetry Anomaly Detection 



This repository contains the code and generated models for classifying robot telemetry data into different operational states: Normal, DoS (Denial of Service), and Malfunction. The pipeline includes robust preprocessing, hyperparameter optimization, and extensive Explainable AI (XAI) analysis using SHAP and LIME.

🚀 Instructions to Run Code



The provided code is designed to be run in a Google Colab environment, which offers the necessary dependencies and GPU/TPU access for the deep learning models and hyperparameter search.

Prerequisites
Google Drive Access: The code relies on reading data files from a mounted Google Drive folder.

Dataset Files: Ensure your dataset files (Normal1.csv, Dos1.csv, etc.) are placed in a designated folder in your Google Drive, matching the path in the code:


DATASET_PATH = '/content/drive/MyDrive/' # or a subfolder like '/content/drive/MyDrive/Telemetry_Data/'
Step-by-Step Execution
Open in Colab: Upload the .ipynb notebook to Google Drive or GitHub and open it in Google Colab.

Install Libraries: Run the first code cell to install the necessary libraries (keras-tuner, lime). The -q flag suppresses verbose output.


!pip install -q keras-tuner
!pip install lime
Mount Drive: Run the next cell to mount your Google Drive. Follow the instructions and grant permission.


from google.colab import drive
drive.mount('/content/drive')
Full Execution: Run the remaining cells sequentially. The notebook executes the following main stages:

Data Loading & Initial Prep: Loads and labels the data, performs forward-filling and scaling.

Model Training & Tuning (CNN & FNN): Uses keras-tuner.RandomSearch to find optimal hyperparameters for the 1D-CNN and FNN models. This step involves multiple trials and takes the longest time.

Model Training & Tuning (XGBoost): Uses sklearn.model_selection.RandomizedSearchCV to tune the XGBoost model.

Evaluation: Prints classification reports and confusion matrices for all three models.

XAI Analysis: Retrains simplified versions of the best XGBoost and FNN models on the clean data and generates SHAP and LIME explanations.

Artifact Saving: The final cells save the trained models (.json, .keras) and preprocessing tools (.pkl) into a local directory named saved_models/.

📦 Required Libraries and Versions



The core dependencies are managed by pip install commands within the notebook. It is recommended to run this code in a fresh environment (like Google Colab) to ensure compatibility. The main dependencies are:

Library	Version (Recommended/Used)	Purpose

1. Python	3.8+	Core Execution Environment
2. TensorFlow	2.x (2.12+)	Deep Learning (FNN, 1D-CNN)
3. Keras-Tuner	Any recent version	Hyperparameter Optimization
4. XGBoost	Any recent version	Gradient Boosting Model
5. NumPy	Any recent version	Numerical Operations
6. Pandas	Any recent version	Data Manipulation
7. Scikit-learn	Any recent version	Preprocessing, Splitting, Evaluation
8. SHAP	Any recent version	SHAP Explanations
9. LIME	Any recent version	LIME Explanations
10. Matplotlib / Seaborn	Any recent version	Plotting and Visualization


⏱️ Expected Runtime for Training



The majority of the runtime is consumed by the hyperparameter tuning steps for the Neural Networks (FNN and 1D-CNN).


1. Model / Step	Description	Estimated Runtime (Colab Standard GPU/CPU)
   
2. 1D-CNN Tuning	kt.RandomSearch with 10 trials (10 epochs max, validation split, early stopping).	15 - 30 minutes

   
3. FNN Tuning	kt.RandomSearch with 10 trials (15 epochs max, validation split, early stopping).	10 - 20 minutes

   
4. XGBoost Tuning	RandomizedSearchCV with 10 iterations (CV=3, n_jobs=-1).	5 - 15 minutes

   
5. XAI SHAP/LIME	Generating SHAP (KernelExplainer for FNN is slow) and LIME explanations for 100 samples and 1 instance, respectively.	20 - 40 minutes (KernelSHAP is compute-intensive)

   
6. Total Estimated Runtime		50 - 105 minutes (Approx. 1 to 1.75 hours)



Note: The runtime is highly dependent on the Colab instance's allocated resources (CPU/GPU) and the size of your dataset (which is implicitly quite large, given the telemetry nature).
