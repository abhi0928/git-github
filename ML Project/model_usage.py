from joblib import load
import numpy as np 

model = load('dragon.joblib')

features = np.array([[0.04932,33,2.18,0,0.472,6.849,70.3,3.1827,7,222,18.4,396.9,7.53,28.2]])

print(model.predict(features))       