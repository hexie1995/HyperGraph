
	
data-load: 
	sbatch run-data-load-test.sh

asymptotics: 
	sbatch run-asymptotics.sh

asymptotics-fig: 
	~/.conda/envs/hypermech/bin/python code/asymptotics.py