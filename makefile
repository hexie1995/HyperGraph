
	
data-load: 
	sbatch run-data-load-test.sh

asymptotics: 
	sbatch run-asymptotics.sh

asymptotics-fig: 
	~/.conda/envs/hypermech/bin/python code/asymptotics.py

all: fig2 fig3 fig4 fig5

fig2: run_fig2_a run_fig2_b
	@echo "Generating fig 2"

fig3: RW_SEM run_fig3
	@echo "Generating fig 3"

fig4: RW_SEM run_fig4_a run_fig4_b run_fig4_c run_fig4_d
	@echo "Generating fig 4"

fig5: RW_LP run_fig5
	@echo "Generating fig 5"

run_fig2_a:
	@echo "Running Fig2_calculate_properties_cluster_part.py"
	python code/Fig2_calculate_properties_cluster_part.py

run_fig2_b: run_fig2_a
	@echo "Running Fig2_Hypergraph_Properties_plotting_part.py"
	python code/Fig2_Hypergraph_Properties_plotting_part.py

# Target to REALWORD SEM CALCULATION
RW_SEM: 
	@echo "Running REAL_WORLD_SEM.py"
	python code/REAL_WORLD_SEM.py

run_fig3: RW_SEM
	@echo "Running Fig3-degree-and-size-distributions.py"
	python code/Fig3-degree-and-size-distributions.py

run_fig4_a: RW_SEM
	@echo "Running simulation_real_world.py"
	python code/simulation_real_world.py

run_fig4_b: RW_SEM run_fig4_a
	@echo "Running simulation_real_world_recovered.py"
	python code/simulation_real_world_recovered.py

run_fig4_c: RW_SEM run_fig4_a run_fig4_b
	@echo "Running get_results_real_world.py"
	python code/get_results_real_world.py

run_fig4_d: RW_SEM run_fig4_a run_fig4_b run_fig4_c
	@echo "Running Fig4_edge_intersection_comparison_colorscheme.py"
	python code/Fig4_edge_intersection_comparison_colorscheme.py

RW_LP: 
	@echo "Running LP-REAL-WORLD.py"
	python code/LP-REAL-WORLD.py

run_fig5: RW_LP
	@echo "Running Fig5-ScatterPlot-Table1and2.py"
	python code/Fig5-ScatterPlot-Table1and2.py
