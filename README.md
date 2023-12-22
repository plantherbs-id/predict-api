# Predict API 
## endpoint of the prediction model

This is python code, make sure you have python installed on your computer or system.

1. Clone the repository then open it using your code editor.
2. Make sure the model file (here named `my_model_fix.h5`), credential file in json format (here named `plantherbs-credential.json`), and requirements file (here named `requirements.txt`) are in the repository.
3. This code is connected to Google Cloud Storage, make sure you have your own GCS bucket (here named `plantherbs-bucket`).
4. Go to the `main.py` file and look at line 34. Make sure the bucket name is `plantherbs-bucker`, or matches the bucket created earlier.
5. Open a terminal in the project root directory, then install the dependencies with the command `pip install -r reqirements.txt`.
6. Run the app using the command `python main.py`.
7. Automatically, the server will run on port 5000.
