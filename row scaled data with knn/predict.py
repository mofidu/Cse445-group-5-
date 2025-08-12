import joblib
import numpy as np
from PIL import Image


model = joblib.load("row_data_tuned_scaled_model.pkl")

THRESHOLD = 5147808.89


label_names = ['Alian_1', 'Elon_musk_5', 'Fida_2', 'Mashrafi_8', 'Messi_3',
               'Mofidul_0', 'Sakib_9', 'Shundar_pichai_4', 'Zayan_7', 'Zuhayr_6']


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  
    img = img.resize((100, 100))                 
    img_array = np.array(img).flatten()          
    return img_array.reshape(1, -1) 

def predict_with_unknown(model, input_data, label_names, threshold):
    knn = model.best_estimator_.named_steps['knn']
    distance, _ = knn.kneighbors(input_data, n_neighbors=1)

    if distance[0][0] > threshold:
        return "Unknown"
    else:
        label = model.predict(input_data)[0]
        return label_names[label]


image_path = "test6.jpg"  
X_input = preprocess_image(image_path)
result = predict_with_unknown(model, X_input, label_names, THRESHOLD)

print("Prediction:", result)
