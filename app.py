from flask import Flask, jsonify, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/')
def price_predict():
    model = pickle.load(open('model.pickle', 'rb'))
    age = request.args.get('Age')
    experience = request.args.get('Experience')
    income = request.args.get('Income')
    zip_code = request.args.get('Zip_code')
    family = request.args.get('Family')
    cCAvg = request.args.get('CCAvg')
    education = request.args.get('Education')

    test_df = pd.DataFrame(
        {'Age': [age], 'Experience': [experience], 'Income': [income], 'Zip_code': [zip_code], 'Family': [family],
         'CCAvg': [cCAvg],
         'Education': [education]})

    predict = model.predict(test_df)

    return jsonify({'Loan approved': str(predict)})


if __name__ == '__main__':
    app.run(debug=True)
