from joblib import dump, load
import numpy as np

model= load('RealEstatePricePredictor.joblib')

features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
                      -11.44443979304, -49.31238772, 7.61111401, -26.0016879, -0.5778192,
                      -0.97491834, 0.41164221, -66.86091034]])

prediction=model.predict(features)

print(prediction)