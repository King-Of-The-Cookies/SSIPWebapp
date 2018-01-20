#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:29:04 2017

@author: guysimons
"""

import os
from flask import Flask, request, render_template, redirect
import numpy as np
import pandas as pd
from scipy import spatial

os.chdir("/Users/lisaherzog/Google Drive/UM/Smart Services/Smart Service Project/WebApplication/SSIPWebapp/webapp")

"""
The function below computes the similarity between all columns in a dataframe. It first creates an empty dataframe to hold the 
similarity values. Then, it computes the similarity between all columns in a nested loop and stores the values in the empty dataframe.
Lastly the index and column names are set and the dataframe containing the similarity values is returned.
"""

def ComputeCosSimilarityMatrix(inputDf):
     similarities = pd.DataFrame(np.zeros((inputDf.shape[1],inputDf.shape[1])))
     for i in range(0, inputDf.shape[1]):
          for j in range(0, inputDf.shape[1]):
               cosSim = 1 - spatial.distance.cosine(inputDf.iloc[:,i], inputDf.iloc[:,j])
               similarities.iloc[i,j] = cosSim
     similarities.columns = inputDf.columns
     similarities.index = inputDf.columns
     return similarities

"""
The purpose of the Predict function is to make a prediction on what activity a the user will like next. It does this based on the similarity matrix and the userVector. The
userVector is a vector that records previous likes(1), dislikes(-1), and not rated (0) values. When the function is called it first checks if the userVector only contains 0s. If this is the case,
it suggests a random activity as no predictions can be made for users that haven't rated anything. If the userVector does contain likes and dislikes
the activities for which rating already exist are stored. Then, the userVector is multiplied with the similarity matrix and the resulting values are divided by the sum of the similarities.
Subsequently, activities that are already rated are removed and the activity with the highest score is returned.
"""

def Predict(userVector, similarityMatrix):
     if userVector.sum()==0:
          randomActivity = np.random.randint(1, similarityMatrix.columns.shape[0])
          return similarityMatrix.columns.values[randomActivity]
     else:
          userVector = userVector.fillna(0)
          previouslyRated = userVector[(userVector > 0) | (userVector < 0)].index.values
          activityScores = similarityMatrix.dot(userVector.values.reshape((similarityMatrix.shape[1],))).div(similarityMatrix.sum(axis=1))
          activityScores = activityScores.drop(previouslyRated)          
     return np.argmax(activityScores)

"""
The userItemMatrix is a matrix of 1 (like) and -1 (dislike) values that is used to compute the similarity matrix. 
"""

userItemMatrix = pd.DataFrame(np.random.choice([1,-1], size = (1000,10)), columns = ["Activity1", "Activity2", "Activity3", "Activity4", "Activity5", "Activity6", "Activity7","Activity8","Activity9","Activity10"])

"""
The next step is to normalize the user vectors in the userItemMatrix. The purpose of this is to make sure that users with many ratings (those that like everything), contribute less to any individual rating. 
To do this we square and sum the items in the user vector, to then take the square root which results in the magintude. 
Then, all user values are divided by the respective user's magnitude. 
"""

magnitude = np.sqrt(np.square(userItemMatrix[userItemMatrix==1]).sum(axis=1))
userItemMatrix[userItemMatrix==1] = userItemMatrix[userItemMatrix==1].divide(magnitude, axis='index')
userItemMatrix = userItemMatrix.fillna(0)

"""
Call the ComputeCosSimilarityMatrix function with the userItemMatrix as input.
"""

similarityMatrix = ComputeCosSimilarityMatrix(userItemMatrix)

############WEB APPLICATION ROUTING##################
app = Flask(__name__)
rating = None
currentActivity = None


"""
Generate new userVector for a user that opens the application for the first time.
"""

userVector = pd.Series(np.zeros((similarityMatrix.shape[1])), index=similarityMatrix.columns.values)

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/A1-food")
def A1food():
    return render_template("A1-food.html")

@app.route("/chooseactivity")
def chooseactivity():
     
     """
     Call Predict function.
     """
     
     predictedactivity = Predict(userVector ,similarityMatrix)
     global currentActivity
     currentActivity = predictedactivity
     return render_template("chooseactivity.html", predictedactivity = predictedactivity)

@app.route('/submitForm', methods=['POST'])
def catchResponse():
     target = request.form['likedActivity']
     global rating
     rating = int(target)
     global currentActivity
     global userVector
     userVector[currentActivity] = rating
     print(userVector)            
     return redirect('/chooseactivity')

if __name__ == "__main__":
    app.run(port = 5001)


