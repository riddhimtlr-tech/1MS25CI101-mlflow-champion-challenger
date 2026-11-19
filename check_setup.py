import sys
import mlflow
import sklearn
import pandas
import numpy

print("Python:", sys.executable)
print("MLflow:", mlflow.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pandas:", pandas.__version__)
print("NumPy:", numpy.__version__)

print("SETUP SUCCESSFUL!")