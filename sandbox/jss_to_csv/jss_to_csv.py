import csv
import re
import numpy as np

class JSS_TO_CSV():
	

	def __init__(self, jss_file):
		
		self.jss_file = jss_file
		self.jss_data = jss_file.read()
		self.csv_data = ''
		
		self.g_order_nr = lambda x,y: int(x.split(" ")[y])		

	def convert(self):
		data = self.clean()	
		
		j = []
		for x in range(len(data)):
			j.append([])

		for z in range(len(data)):
			x = data[z].split()
			for i in range(1, len(x), 2):
				j[z].append("{} {}".format(x[i-1], x[i]))
		
	
		for x in range(len(j)):
			aux = 0	
			for y in range(len(j[x])):
				for k in range(len(j[x])):	
					if self.g_order_nr(j[x][y], 0) < self.g_order_nr(j[x][k], 0):
						aux = j[x][y]
						j[x][y] = j[x][k]
						j[x][k] = aux
						

		self.csv_data = data = self.chg_order(j)
		return data
	

	def to_csv(self):
		# writing to csv file 
		with open('data/data.csv', 'w') as csvfile: 
			# creating a csv writer object 
			csvwriter = csv.writer(csvfile) 

			# writing the data rows 
			csvwriter.writerows(self.csv_data)




	def chg_order(self, data):
			
		for x in range(len(data)):
			for y in range(len(data[x])):
				data[x][y] = "{} {}".format(self.g_order_nr(data[x][y], 0)+1, self.g_order_nr(data[x][y], 1))
		
		return data	

 

	def clean(self):
		
		data = re.split('\n',self.jss_data)
		data_cp = data.copy()
	
		c = 0	
		for x in range(len(data)):
			if re.search('#', data_cp[x]) or not data_cp[x]:
				data.pop(x-c)	
				c += 1
		data.pop(0)		

		return data		
	
	def __str__(self):
		return "{}".format(self.jss_data)		



f = open("data/ft06.jss", "r")

obj = JSS_TO_CSV(f)
csv_file = obj.convert()
obj.to_csv()




