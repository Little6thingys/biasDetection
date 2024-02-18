from flask import Flask, render_template, request
import time
import train2
app = Flask(__name__)


@app.route("/")
def main_page():
    return render_template("index.html")

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

@app.route("/load_data", methods=["POST"])
def load_info():
    data = request.form["data"]
    toxic_prediction, toxic_confidence, sentiment_prediction, sentiment_confidence, label_prediction, label_confidence = train2.mainFunc(data)
    if toxic_prediction[0] == 0:
        toxic_confidence = 100 - int(toxic_confidence)
    if sentiment_prediction[0] == 0:
        sentiment_confidence = 100 - int(sentiment_confidence)
    if label_prediction[0] == 'Neutral':
        label_confidence = 100 - int(label_confidence)
    print(label_prediction)
    return {"data": [int(label_confidence), int(toxic_confidence), int(sentiment_confidence)]}

@app.route("/about")
def about():
    return render_template("about.html")



if __name__ == "__main__":
    app.run(port=8000, debug=True)