import re

class jss_data():

	def __init__(self, f_name):
		self.jss_data = open(f_name, 'r').read()


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

	def convert(self):
		c_data = self.clean() 		

		data = {}
		for x in range(len(c_data)):
			p_data = c_data[x].split()
			data['job'+str(x)] = []
		
			for i in range(0, len(p_data), 2):
				data['job'+str(x)].append([int(p_data[i]), int(p_data[i+1])])

		return data

