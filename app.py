from flask import Flask, render_template, request
import pickle
from flask_cors import CORS, cross_origin

app = Flask(__name__)

saved_model = pickle.load(open('model.pkl','rb'))
saved_scaler = pickle.load(open('scaler_trans.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            rm = request.form['rm']
            age = request.form['age']
            ym = request.form['ym']
            child = request.form['child']
            rel = request.form['rel']
            edu = request.form['edu']
            wo = request.form['wo']
            mo = request.form['mo']
            scaled = saved_scaler.transform([[rm,age,ym,child,rel,edu,wo,mo]])
            print(scaled)
            res = saved_model.predict(scaled)
            print(res[0])
            if res[0]==1:
                result = "Yes"
            else:
                result = "No"
            print("result: ",result)
            
            return render_template('result.html',result=result)
        except Exception as e:
            error = {'error': e}
            return render_template('404.html', error=error)
        


if __name__ == "__main__":
    app.run(debug=True)