from os import sep
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')#print(plt.stype.available)
class Constant:
	MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	MAX_STEP_HOURS = 24 * 30
class DataManager():
    def __init__(self) -> None:
        self.PV_Generation=[]
        self.Prices=[]
        self.Electricity_Consumption=[]
    def add_pv_element(self,element):self.PV_Generation.append(element)
    def add_price_element(self,element):self.Prices.append(element)
    def add_electricity_element(self,element):self.Electricity_Consumption.append(element)

    # get current time data based on given month day, and day_time
    def get_pv_data(self,month,day,day_time):return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_price_data(self,month,day,day_time):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_electricity_cons_data(self,month,day,day_time):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
