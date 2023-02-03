import datetime
from datetime import datetime
import pandas as pd

class data_datapreprocessing():
    def __init__(self,df, dropdown):
        self.DF = None      
        self.df = df
        self.dropdown = dropdown
    
    
    def replace_time_date(self):
        pos = 0
        for i in self.df["Date"]:
            count = 0
            res = ''
            for j in i:
                res += j
                count += 1
                if count == 10:
                    res = datetime.strptime(res, '%Y-%m-%d').date()
                    self.df.loc[pos,'Date'] = res
                    break
            pos += 1
            
    def create_new_dataframe(self):
        self.DF = pd.DataFrame()
        lcs = [i for i in self.df['Date']]
        date_time = pd.to_datetime(lcs)
        self.DF['value'] = self.df[self.dropdown]
        return self.DF