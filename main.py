from flask import Flask, flash, jsonify, render_template, redirect, url_for, request
from load_deep_net import main as predict_value
from flaskwebgui import FlaskUI

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
ui = FlaskUI(app)

@app.route("/")
def gui():
    return render_template("gui.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/index')
def index():
    return render_template("interface/index.html")

@app.route('/statestics')
def statestics():
    return render_template("interface/statestics.html")

@app.route('/tables')
def tables():
    return render_template("interface/tables.html")

@app.route('/documentation')
def documentation():
    return render_template("interface/documentation.html")

@app.route('/view_model')
def view_model():
    return render_template("interface/view_model.html")

@app.route('/view_predict')
def view_predict():
    return render_template("interface/view_predict.html")

@app.route('/charts')
def charts():
    return render_template("interface/charts.html")

@app.route('/submit_view_predicted', methods=['POST'])
def submit_view_predicted():
    print("*************** Predecting... **************")
    age = request.form['age']
    year = request.form['year']
    node = request.form['node']

    input = [(int(age), int(year), int(node))]
    val = predict_value(input)

    if(val == 0):
        rep = 'survive 5-years or longer incha\'ALLAH'
        flash("survive 5-years or longer incha\'ALLAH")
    else:
        rep = 'dead within 5-year REBI yrehmou'
        flash("dead within 5-year REBI yrehmou")

    data = {
        'age': age,
        'year': year,
        'node': node,
        'val': rep
        }

    print('------------------',val,'----------------------')

    return redirect("/view_predict")


if __name__ == "__main__":
    app.run(debug=True)
    #ui.run()
