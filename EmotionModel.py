# Import Packages
import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import glob 
import pickle

def UserInput(userInputImage):
    
    # convert to gray scale
    img = "/Users/sanzi/Desktop/Master_Project/sanzida.png"
    Image.open(userInputImage).convert('L').save(img)
    
    # Run Open Face 
    os.system("/Users/sanzi/Desktop/Master_Project/openFace/OpenFace/build/bin/FeatureExtraction -f {}".format(img))
    useroutputCSVpath = "/Users/sanzi/Desktop/Facial-Recognition-Music-Recommendation-System/processed"
    csvoutput = glob.glob(useroutputCSVpath + "/*.csv")
    
    return csvoutput
    
    
def CleanModelCSV(inputCSV):
    
    # Columns needed from the CSV files
    csv = pd.read_csv(inputCSV)    
    csv = csv[['AU01_c', 'AU02_c','AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c',
               'AU14_c','AU15_c', 'AU17_c','AU20_c', 'AU23_c','AU25_c','AU26_c', 'AU28_c','AU45_c', 
               'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r','AU06_r','AU07_r', 'AU09_r', 'AU10_r',
               'AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']]
    
    return csv


def TreeModel(inputCSV):
    
    # Decision Tree Model
    filename = "/Users/sanzi/Desktop/Facial-Recognition-Music-Recommendation-System/joblib_model.sav"
    # load model from pickle file
    loaded_model = joblib.load(filename)

    # evaluate model 
    y_predict = loaded_model.predict(inputCSV)
    
    return y_predict



def main():
    file = "/Users/sanzi/Desktop/example.png"
    x = "/Users/sanzi/Desktop/Facial-Recognition-Music-Recommendation-System/emotion.csv"
    y = "/Users/sanzi/Desktop/Facial-Recognition-Music-Recommendation-System/emotion_y.csv"
    inputCSV = UserInput(file)
    inputCSV = inputCSV[0]
    icsv = CleanModelCSV(inputCSV)
    #print(icsv)
    predicted = TreeModel(icsv)
    emotion_list = {1: "Happy", 2: "Netural", 3:"Sad" , 4: "Anger" }
    predicted = emotion_list.get(predicted)
    print("Predicted Value is: ")
    print(predicted)

if __name__ == "__main__":
    main()