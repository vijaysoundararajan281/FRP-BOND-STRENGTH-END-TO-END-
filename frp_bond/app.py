from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle as pk 
import xgboost


app = Flask(__name__)
model = pk.load(open('model.pkl', 'rb'))

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def pred():
    intinp = [float(x) for x in request.form.values()]
    inp = [np.array(intinp)]
    a = pd.DataFrame(inp, columns=['(mm)', '(Mpa)', '(Gpa)', '(mm).1', '(mm).2', '(mm).3'])
    res = model.predict(a)

    op = round(res[0], 2)

    return render_template('index.html', pred_text="The Bond Strength is {}".format(op))



if __name__ == '__main__':
    app.run(debug=True)