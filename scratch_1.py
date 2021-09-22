from flask import Flask, request, render_template
import numpy as np
import pickle

# %%
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# %%


@app.route('/')
def home():
    return render_template('index.html')


# %%
@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    :return:
    """

    int_feature = []
    for x in request.form.values():
        if x != '':
            int_feature.append(x)

    final_feature = [np.array(int_feature)]
    prediction = model.predict(final_feature)

    return render_template('index.html', prediction_text='Loan should be {0}'.format(prediction[0]))


if __name__ == '__main__':
    app.run(port=5555, debug=True)
