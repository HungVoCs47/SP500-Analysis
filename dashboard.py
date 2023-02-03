import streamlit as st  
import yfinance as yf
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
from dataloader_streamlit import SP500DataLoader
from data_preprocess import data_datapreprocessing
import numpy as np
import torch
import databases as db
import pyodbc
import csv
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from load_data import load_data_all
from LSTM import LSTM
# connect database to continue to train pretrained model
scalar = StandardScaler()

st.title('SP500 FINANCE DASHBOARD')

#tickers = ("A", "AAL", "AAP", "AAPL", "ABC" )
#dropdown = st.multiselect('Pick your assets', tickers)


start = st.date_input('Start', value = pd.to_datetime('2011-01-01'))
end = st.date_input('End', value = pd.to_datetime('today'))
start_1 = '2011-01-01'
end_1 = '2021-01-31'

dropdown = st.text_input('Enter Stock Ticker', 'A')
N = st.number_input('Enter number of related Stock Ticker', 5)

data_downloader_object = SP500DataLoader(dropdown)



if len(dropdown) > 0:
    df = data_downloader_object.get_cleaned_returns(
        start_date=(2011, 1, 1), end_date=(2023, 1, 31),
        interval='1d', column='Adj Close', 
        save_as_h5=False, save_as_csv=True
    )


    st.header('Returns of stock {}'.format(dropdown))
    st.subheader('Data from 2010 - 2023')
    st.line_chart(df[dropdown])
    st.title('Stock Trend Prediction')
    df = pd.read_csv('Data\S&P500-cleaned_returns.csv')
    #st.subheader('Data from 2010 - 2021')
    #st.write(df.describe())
    #with open('/content/Data/S&P500-cleaned_returns.csv') as csv_file:
    #    csv_reader = csv.reader(csv_file, delimiter = ',')
    #   list_of_column_names = []
    #for row in csv_reader: 
    #   list_of_column_names.append(row)
    #    break
    df1 = scalar.fit_transform((np.array(df[dropdown])).reshape(-1,1))
    data_all_train, data_all_test = load_data_all(df1, look_back=32)
    data_all_train = torch.from_numpy(data_all_train).type(torch.Tensor) 
    data_all_test = torch.from_numpy(data_all_test).type(torch.Tensor)
    model = LSTM(input_dim = 1, hidden_dim = 32, num_layers = 3, output_dim = 1)
    model.load_state_dict(torch.load('pretrain\LSTM_1400.pt'))
    y_data_all = model(data_all_train)
    y_data_all = scalar.inverse_transform(y_data_all.detach().numpy())
    data_all_test = scalar.inverse_transform(data_all_test.detach().numpy())
    st.subheader('Predicting Stock Price of {} with Time Chart'.format(dropdown))
    #st.subheader('Number of related companies of {} with Time Chart'.format(dropdown))
    conn = pyodbc.connect(
    Trusted_Connection = "Yes",
    Driver = '{ODBC Driver 17 for SQL Server}',
    Server = "HUNGVO",
    database = "test"
    )
    cursor = conn.cursor()
    database = pd.read_csv('Data\S&P500-cleaned_returns.csv')
    cursor.execute("CREATE TABLE stock (id DATETIME PRIMARY KEY, stock float)")
    cursor.commit()
    da_ = data_datapreprocessing(df = database, dropdown= database.columns[1])
    da_.replace_time_date()
    for row in database.itertuples():
        cursor.execute('''INSERT INTO test.dbo.stock (id, stock) VALUES (?,?)''', row.Date, row.A)
    cursor.commit()

    st.subheader('Predicted data from 2011 - 2023')
    chart_data = pd.DataFrame(
    {'Real': data_all_test.reshape(-1),
     'Predicted': y_data_all.reshape(-1)
    })
    
    st.line_chart(chart_data)
    
    st.subheader('Most related tickers')
    with open('Data\S&P500-cleaned_returns_all.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        list_of_column_names = []
        for row in csv_reader: 
            list_of_column_names.append(row)
            break
    new_database =  pd.read_csv('Data\S&P500-cleaned_returns_all.csv')
    corre = {}
    for i in list_of_column_names[0]:
        if i == 'Date':
            continue
        if i != 'A':
            corre[i] = new_database[i].corr(new_database['A'])
    sorted_corre = sorted(corre.items(), key=lambda x:x[1])
    
    headers = []  
    for i in range(1,6):
        headers.append(sorted_corre[-i][0])
    headers.append('A')
    trace = list(new_database.loc[:, i] for i in headers)
    #trace.append(new_database.loc[:, 'A'])
    #temp =  list_of_column_names[0:5]
    #for ticker in temp:
    #    trace.append(new_database.loc[:, ticker])
    #    headers.append(ticker)
    pct_frame = pd.concat(trace, axis = 1, keys = headers)
    corr_frame = pct_frame.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_frame,annot=True, ax = ax)
    st.write(fig)
    # After use remember to delete all database
    
    
    
    
    
    