Human Activity Recognition using SVM
This project demonstrates the implementation of Human Activity Recognition (HAR) using Support Vector Machine (SVM). The goal is to classify various human activities based on data from accelerometers or gyroscopes. HAR has applications in healthcare, sports, and smart devices.

Dataset
The dataset used for this project contains time-series data on various human activities, such as walking, sitting, standing, and lying. This data is gathered using wearable devices equipped with sensors like accelerometers and gyroscopes.

The dataset can be obtained from the UCI Machine Learning Repository or other sources providing human activity data.
Features
Preprocessing: The data is cleaned, and features are extracted before applying machine learning models.
Dimensionality Reduction: Techniques like PCA are used to reduce the dimensionality of the dataset.
Classification using SVM: SVM is employed to classify activities based on sensor data.
Model Evaluation: Performance of the model is evaluated using metrics such as accuracy, precision, recall, and F1-score.
Dependencies
To run this project, you'll need the following Python libraries:

numpy
pandas
matplotlib
scikit-learn
You can install the dependencies using:

bash
Copy code
pip install -r requirements.txt
How to Run
Clone this repository:
bash
Copy code
git clone https://github.com/anomalinine/Human-Activity-Recognition-using-SVM.git
Navigate to the project directory:
bash
Copy code
cd Human-Activity-Recognition-using-SVM
Open the Jupyter Notebook:
bash
Copy code
jupyter notebook Human\ Activity\ Recognition.ipynb
Follow the steps in the notebook to load the dataset, preprocess the data, and train the SVM model.
Results
The model achieves an accuracy of XX% on the test dataset. Below are some key performance metrics:

Future Work
Implement other machine learning models such as Random Forest, K-Nearest Neighbors, or Neural Networks for comparison.
Experiment with deep learning techniques using LSTM or CNN for sequential data.
Improve feature engineering to enhance model performance.
License
This project is licensed under the MIT License. Feel free to use, modify, and distribute this code for your projects.

