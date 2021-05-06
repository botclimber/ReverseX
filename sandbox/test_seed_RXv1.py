import random

JOBS = 3
MACHINES = 3


def seed():
	''' 
	Generate data for training
	'''
	machines = [x for x in range(MACHINES)]     
	data = {}
	    
	for x in range(JOBS):
		data['job'+str(x)] = []
		    
		random.shuffle(machines)
		for j in machines:
			data['job'+str(x)].append([ j, random.randrange(0, 10)])

	return data

def ini_state(data):
	''' 
	Generate initial state  
	'''
	state = []

	for x in range(MACHINES):
		state.append(0)

	for y in range(JOBS):
		state.append(data['job'+str(y)][0][0])
		state.append(data['job'+str(y)][0][1])
		state.append(len(data['job'+str(y)]))
	state.append(0)

	return state


if __name__ == "__main__":

	data = seed()
	print(ini_state(data))
