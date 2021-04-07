# AI RL project

Educational project

JOB SHOP Problem (JSP) . The point is to make an RL Agent solve scheduling problems for us.

requirement:
python==(3.5 > < 3.7)


- environment dir: RX_env
- agents dir: RX_agent

## Issues
v0.14:
- Testing Agents and alg
	- Agent from stable baselines
		- dqn
		- a2c 

situation point:

after a few tests with different algorithms (dqn, a2c, acer) the conclusion is that a2c in this particular problem acts better, and gives better results.

dqn: with 500000 steps:
- gives a range of results with huge disparity.
- train takes to much time 


a2c: with 500000 steps:
- gives a range of results with less disparity all the solutions are optimal.
- train faster than dqn

Train value_loss parcial variance:

![Image of api_doc](https://github.com/botclimber/ReverseX/blob/master/img/a2c_eval.png)

Evalutation:

![Image of api_doc](https://github.com/botclimber/ReverseX/blob/master/img/a2c_train.png)


v0.13:
- random seed training 
- infinite action bug fixed

v0.12:
- BUG:
	- infinite action

v0.11:-


## TODO:

- [x] aply changes from sandbox to main file;
- [x] make input dinamic;
- [x] change process order input;

- test AI Agents:
	- [x] Stable Baselines
	- [ ] PARL


