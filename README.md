# S&P500-Analysis
![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Frontend](	https://img.shields.io/badge/MySQL-005C84?style=for-the-badge&logo=mysql&logoColor=white)
![Frontend](https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white)
![Frontend](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

A system that predict stock price trend of SP500 companies using deep learning architecture.

## Overview
The stock price is predicted base on the ticker which user entered. Furthermore, the system returns the most N affected stock price which user entered. The detail of the data is dowloaded from http://en.wikipedia.org/wiki/List_of_S%26P_500_companies

## How to run the project?
1) Clone or download this repository to your local machine.
2) Install all the libraries mentioned in the requirements.txt file with the command pip install -r requirements.txt
3) Open your terminal/command prompt from your project directory and run the file `dashboard.py` by executing the command `streamlit run dashboard.py`
4) Go to your browser and type http://localhost:8501 in the address bar.

## Architecture
![image](https://github.com/HungVoCs47/SP500-Analysis/blob/main/image/Screenshot%202023-02-03%20154744.png)



## Pretrain model and dataset
|   Model  | Download |
| -------- | -------- |
| LSTM-SMALL    | [LSTM_small_prediction.pt](https://github.com/HungVoCs47/SP500-Analysis/blob/main/pretrain/LSTM_1400.pt) |

|   Dataset  | Download |
| --------| -------- |
| Stock     | [S&P500.csv](https://github.com/HungVoCs47/SP500-Analysis/blob/main/data/S%26P500-cleaned_returns_all.csv) |

## An Finetuning example
Please see detales in [pic_inference.py](https://github.com/HungVoCs47/SP500-Analysis/blob/main/test.ipynb)
