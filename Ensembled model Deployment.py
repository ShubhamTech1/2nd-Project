from flask import Flask, render_template, request
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote 
import pickle
import joblib

DT_best = pickle.load(open('decisiontree.pkl','rb'))
rf_best = pickle.load(open('GSCV_RF.pkl','rb'))

clean = joblib.load('clean_DT')


# MySQL Database connection
user = 'user1' # user name
pw = 'user1' # password
db = 'sales_db' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

def decision_tree(data_new):
    clean1 = pd.DataFrame(clean.transform(data_new), columns = clean.get_feature_names_out())
    
    prediction = pd.DataFrame(DT_best.predict(clean1), columns = ['DT_Sales'])                                                                                   
    prediction2 = pd.DataFrame(rf_best.predict(clean1), columns = ['RF_Sales'])
                      
    final_data = pd.concat([prediction, prediction2, data_new], axis = 1)
    return(final_data)
            
#define flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data_new = pd.read_csv(f)
       
        final_data = decision_tree(data_new)
        
        final_data.to_sql('ClothCompany_Data_Prediction'.lower(), con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        
        html_table = final_data.to_html(classes='table table-striped')
                          
        return render_template("new.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #8f6b39;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #32b8b8;\
                    }}\
                            .table tbody th {{\
                            background-color: #3f398f;\
                        }}\
                </style>\
                {html_table}") 

if __name__=='__main__':
    app.run(debug = True)
