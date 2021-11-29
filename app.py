from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os
#from flask_cors import CORS, cross_origin
# import flask_monitoring.dashboard as dashboard

'''Load pickel file'''
file = os.listdir('./bestmodel/')[0]
model = pickle.load(open('./bestmodel/'+file, 'rb'))
scaler = pickle.load(open('standard_scaler.pkl','rb'))

app = Flask(__name__)
# dashboard.bind(app)
# CORS(app)

@app.route('/')
#@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
#@cross_origin()
def predict():
    if request.method == 'POST':
        Item_Identifier = request.form['Item ID']
        Item_Weight = float(request.form['Weight'])    ansiwrap==0.8.4
        item_fat_content = request.form['Item Fat Conteasync-generator==1.10nt']
                                                       black==21.5b0
        if (item_fat_content == 'low'):                blinker==1.4
            item_fat_content = 0, 0                    certifi==2021.5.30
        elif (item_fat_content == 'reg'):              click==8.0.1
            item_fat_content = 0, 1                    colorama==0.4.4
        else:                                          colorlover==0.3.0
            item_fat_content = 1, 0                    cufflinks==0.17.3
                                                       Flask==2.0.1
        Item_Fat_Content_1, Item_Fat_Content_2 = item_fgunicorn==20.1.0at_content
                                                       imbalanced-learn==0.8.0
        Item_Visibility = float(request.form['Range 0.1imblearn==0.0-2.0'])
                                                       itsdangerous==2.0.1
        Item_MRP = float(request.form['Item MRP'])     Jinja2==3.0.1
                                                       joblib==1.0.1
        Outlet_Identifier = request.form['Outlet ID']  jupyterlab-pygments==0.1.2
                                                       lazypredict==0.2.9
        Outlet_ID = Outlet_Identifier                  lightgbm==2.3.1
        if (Outlet_Identifier == 'OUT013'):            llvmlite==0.34.0
            Outlet_Identifier = 1, 0, 0, 0, 0, 0, 0, 0,MarkupSafe==2.0.1 0
        elif (Outlet_Identifier == 'OUT017'):          mypy-extensions==0.4.3
            Outlet_Identifier = 0, 1, 0, 0, 0, 0, 0, 0,nbclient==0.5.3 0
        elif (Outlet_Identifier == 'OUT018'):          nbconvert==6.0.7
            Outlet_Identifier = 0, 0, 1, 0, 0, 0, 0, 0,numba==0.51.2 0
        elif (Outlet_Identifier == 'OUT019'):          numpy==1.21.1
            Outlet_Identifier = 0, 0, 0, 1, 0, 0, 0, 0,pandas==1.3.1 0
        elif (Outlet_Identifier == 'OUT027'):          pandas-visual-analysis==0.0.4
            Outlet_Identifier = 0, 0, 0, 0, 1, 0, 0, 0,papermill==2.3.2 0
        elif (Outlet_Identifier == 'OUT035'):          pathspec==0.8.1
            Outlet_Identifier = 0, 0, 0, 0, 0, 1, 0, 0,plotly==4.14.3 0
        elif (Outlet_Identifier == 'OUT045'):          python-dateutil==2.8.2
            Outlet_Identifier = 0, 0, 0, 0, 0, 0, 1, 0,pytz==2021.1 0
        elif (Outlet_Identifier == 'OUT046'):          retrying==1.3.3
            Outlet_Identifier = 0, 0, 0, 0, 0, 0, 0, 1,scikit-learn==0.24.2 0
        elif (Outlet_Identifier == 'OUT049'):          scipy==1.7.1
            Outlet_Identifier = 0, 0, 0, 0, 0, 0, 0, 0,shap==0.37.0 1
        else:                                          shapash==1.3.2
            Outlet_Identifier = 0, 0, 0, 0, 0, 0, 0, 0,six==1.16.0 0
                                                       slicer==0.0.3
        Outlet_1, Outlet_2, Outlet_3, Outlet_4, Outlet_tenacity==7.0.05, Outlet_6, Outlet_7, Outlet_8, Outlet_9 = Outlet_Identifier
                                                       textwrap3==0.9.2
        Outlet_Year = int(2021 - int(request.form['Yearthreadpoolctl==2.2.0']))
                                                       tqdm==4.56.0
        Outlet_Size = request.form['Size']             waitress==2.0.0
        if (Outlet_Size == 'Medium'):                  Werkzeug==2.0.1
            Outlet_Size = 1, 0                         wincertstore==0.2
        elif (Outlet_Size == 'Small'):
            Outlet_Size = 0, 1
        else:
            Outlet_Size = 0, 0

        Outlet_Size_1, Outlet_Size_2 = Outlet_Size

        Outlet_Location_Type = request.form['Location Type']
        if (Outlet_Location_Type == 'Tier 2'):
            Outlet_Location_Type = 1, 0
        elif (Outlet_Location_Type == 'Tier 3'):
            Outlet_Location_Type = 0, 1
        else:
            Outlet_Location_Type = 0, 0

        Outlet_Location_Type_1, Outlet_Location_Type_2 = Outlet_Location_Type

        Outlet_Type = request.form['Outlet Type']
        if (Outlet_Type == 'Supermarket Type1'):
            Outlet_Type = 1, 0, 0
        elif (Outlet_Type == 'Supermarket Type2'):
            Outlet_Type = 0, 1, 0
        elif (Outlet_Type == 'Supermarket Type3'):
            Outlet_Type = 0, 0, 1
        else:
            Outlet_Type = 0, 0, 0

        Outlet_Type_1, Outlet_Type_2, Outlet_Type_3 = Outlet_Type

        Item_Type_Combined = request.form['Item Type']

        if (Item_Type_Combined == "Food"):
            Item_Type_Combined = 1, 0
        elif (Item_Type_Combined == "Non-consumable"):
            Item_Type_Combined = 0, 1
        else:
            Item_Type_Combined = 0, 0

        Item_Type_Combined_1, Item_Type_Combined_2 = Item_Type_Combined

        data = [Item_Weight, Item_Visibility, Item_MRP, Outlet_Year, Item_Fat_Content_1, Item_Fat_Content_2,
                Outlet_Location_Type_1, Outlet_Location_Type_2, Outlet_Size_1, Outlet_Size_2, Outlet_Type_1, Outlet_Type_2,
                Outlet_Type_3, Item_Type_Combined_1, Item_Type_Combined_2, Outlet_1, Outlet_2, Outlet_3, Outlet_4, Outlet_5,
                Outlet_6, Outlet_7, Outlet_8, Outlet_9]
        features_value = [np.array(data)]

        features_name = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years',
                         'Item_Fat_Content_1', 'Item_Fat_Content_2', 'Outlet_Location_Type_1',
                         'Outlet_Location_Type_2', 'Outlet_Size_1', 'Outlet_Size_2',
                         'Outlet_Type_1', 'Outlet_Type_2', 'Outlet_Type_3',
                         'Item_Type_Combined_1', 'Item_Type_Combined_2', 'Outlet_1', 'Outlet_2',
                         'Outlet_3', 'Outlet_4', 'Outlet_5', 'Outlet_6', 'Outlet_7', 'Outlet_8',
                         'Outlet_9']

        df = pd.DataFrame(features_value, columns=features_name)

        std_data=scaler.transform(df)

        myprd = model.predict(std_data)
        output = round(myprd[0], 2)

        # if output < 0:
        #     return render_template('index.html',prediction_texts=f"Sorry you cannot sell. Sale is negative: {output}")
        # else:
            # return render_template('result.html', prediction_text=f'The Sales production of {Item_Identifier} '
            #                                        f'by '{Outlet_ID} Outlet is  Rs {output}/-')

            # return render_template('index.html',prediction_text="Item_Outlet_Sales at {}".format(output))

        return render_template('result.html',
                                   prediction=output,
                                   Item_Identifier=Item_Identifier,
                                   Outlet_Identifier=Outlet_ID)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    # port = int(os.getenv("PORT"))
    # host = '0.0.0.0'
    # httpd = simple_server.make_server(host=host, port=port, app=app)
    # httpd.serve_forever()

    app.run(debug=True)
