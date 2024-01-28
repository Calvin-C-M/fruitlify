import os
from flask import Flask, render_template, request
from flask_mysqldb import MySQL
from classification import get_prediction
import numpy as np

index_of_classes = {
    0: "Apple Bad",
    1: "Apple Good",
    2: "Banana Bad",
    3: "Banana Good",
    4: "Guava Bad",
    5: "Guava Good",
    6: "Lime Bad",
    7: "Lime Good",
    8: "Orange Bad",
    9: "Orange Good",
    10: "Pomegranate Bad",
    11: "Pomegranate Good",
}

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'fruit_classification'

mysql = MySQL(app)

@app.route("/")
def home() :
    return render_template("index.html")

@app.route("/classification")
def classification() :
    return render_template("classification.html")

@app.route("/classify", methods=['POST'])
def classify() :
    if 'fruit-image' not in request.files :
        return "No image file"
    
    fruit_image = request.files['fruit-image']
    
    filepath = os.path.join("uploads/temp", fruit_image.filename)
    fruit_image.save(filepath)
    
    predictions = get_prediction(filepath)
    predicted_class = np.argmax(predictions[0])
    class_name = index_of_classes.get(predicted_class, "Unknown")
    
    predicted_values = [round(values * 100, 2) for values in predictions[0]]
    
    return render_template("result.html", class_names=index_of_classes, predict=predicted_values, fruit_class=class_name)

@app.route("/feedback", methods=['POST'])
def feedback() :
    form = request.form
    feedback = 1 if form['feedback-validation'] == "yes" else 0
    correct_predict = form['correct-predict']
    
    cur = mysql.connection.cursor()
    cur.execute(f"INSERT INTO feedback(validation, actual) VALUES ({feedback}, '{correct_predict}')")
    mysql.connection.commit()
    cur.close()
    
    return render_template("feedback.html", feed=feedback, correct=correct_predict)

if __name__ == "__main__" :
    app.run(debug=True, host="0.0.0.0", port=5000)