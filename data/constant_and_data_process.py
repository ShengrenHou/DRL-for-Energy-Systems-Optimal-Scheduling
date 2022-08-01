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
# def load_data_tables():
# hourly pv generation data for a year 
pv_df=pd.read_csv('C:/Users/hshengren/Documents/GitHub/unit-commitment-example/generator_battery_env/data/PV.csv',sep=';')
#hourly price data for a year 
price_df=pd.read_csv('C:/Users/hshengren/Documents/GitHub/unit-commitment-example/generator_battery_env/data/Prices.csv',sep=';')
# mins electricity consumption data for a year 
electricity_df=pd.read_csv('C:/Users/hshengren/Documents/GitHub/unit-commitment-example/generator_battery_env/data/H4.csv',sep=';')
pv_data=pv_df['P_PV_'].apply(lambda x: x.replace(',','.')).to_numpy(dtype=float)
price=price_df['Price'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
electricity=electricity_df['Power'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
# netload=electricity-pv_data
data_manager=DataManager()
for element in pv_data:
	data_manager.add_pv_element(element*200)
for element in price:
	element/=10
	if element<=0.5:
		element=0.5
	data_manager.add_price_element(element)
for i in range(0,electricity.shape[0],60):
	element=electricity[i:i+60]
	data_manager.add_electricity_element(sum(element)*300)

# for element in netload:
# 	data_manager.add_netload_element(element)
# print(Constant.MONTHS_LEN[:4-1])
import matplotlib.pyplot as plt
def plot():
	
	pv=data_manager.PV_Generation
	price=data_manager.Prices
	electricity=data_manager.Electricity_Consumption
	fig=plt.figure(figsize=(16,9))
	ax1=fig.add_subplot(311)
	ax1.plot(pv[144:],label='pv_generation')
	ax1.set_ylabel('PV Generation')
	# ax.plot(price,label='electricity price')
	ax2=fig.add_subplot(312)
	ax2.plot(electricity[144:],label='electricity demand')
	ax2.set_ylabel('Electricity Demand')
	#
	ax3=fig.add_subplot(313)
	ax3.plot(price[144:],label='electricity price',drawstyle='steps-post')
	ax3.set_ylabel('Electricity Price')

	plt.xlabel('hours')
	plt.show()
if __name__=='__main__':
	plot()
	pv_array=np.array(data_manager.PV_Generation)
	load_array=np.array(data_manager.Electricity_Consumption)
	price_array=np.array(data_manager.Prices)
	print(pv_array.mean(),pv_array.max(),pv_array.min())
	print(load_array.mean(),load_array.max(),load_array.min())
	print(price_array.mean(),price_array.max(),price_array.min())
	