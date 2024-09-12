# HyperGraph

### Edge-Copying Generative Model and Link Prediction via Stochastic Expectation Maximization in Large Hypergraphs

This is the GitHub repo accompany the [paper by Xie He, Phil Chodrow, Peter Mucha](To be submitted). 
The paper is currently in preparation for submission. 

**Please cite the paper when using the data or code. See License Information for more details on Usage.**


</div>

<h2 align="center">System Requirements </h2>

To reproduce all results from our experiments, you will need:
1. Python 3.10.13
2. [xgi](https://xgi.readthedocs.io/en/stable/) 0.8.2
3.  patsy-0.5.6 statsmodels-0.14.2
You can check these versions respectively with:

```bash
$ python --version
```
```bash
$ import xgi
$ xgi.__version__
```

Note:

1. If you wish to recreate the degree distribution plot, you need to install the code accompany the paper [The simpliciality of higher-order networks](https://github.com/nwlandry/the-simpliciality-of-higher-order-networks) because of the dependency of for edit simplicity and face edit simplicity. Please also cite the original paper if used.
2. The data folder contains NOT xgi dataset (which can be obtained from the xgi package directly), but rather the three bio-medical dataset used in both [Logical Hypergraph Link Prediction](https://github.com/yang1992samantha/LHP) and [Neural Hypergraph Link Prediction](https://malllabiisc.github.io/publications/papers/nhp_cikm20.pdf). We did NOT create these datasets but uploaded these so that readers can conveniently access the information without needing to look them up elsewhere. For clear reference, please see their papers and cite the original sources and datasets when used.

```bash
pip install python==3.10.13 xgi==0.8.2
```

The above environment has been tested to build successfully and run all the following experiments successfully in both Windows and Linux environments. If support needed for Mac, please report in Issues. 


</div>

<h2 align="center">Model Instructions </h2>

Please see `model.py` if you are only interested in using the model. Detailed example usage for generating a synthetic hypergraph can be found in `sem-demo-example-usage.ipynb`


</div>
</div>

<h2 align="center">Figure Generation Instructions </h2>

You can make figures with the makefile. 

```bash
make fig2 fig3 fig4 fig5
```

Or you can see below for detailed instructions. 

1. Figure 1 is genereated as a Toy Model via Google Drawing.
2. Figure 2 is generated via running: a) `Fig2_calculate_properties_cluster_part.py`, b) `Fig2_Hypergraph_Properties_plotting_part.py`. Feel free to check `Fig2-Hypergraph Properties- plotting part.ipynb` for the display of figure.
3. Figure 3 is generated via running: a) `REAL_WORLD_SEM.py`(note the comments for the difference of generation of real-world xgi datasets and the bio-medical datasets, if run correctly, you should see a folder named `sem_results_new` with results file for each xgi dataset), b) `Fig3-degree-and-size-distributions.py`.
4. Figure 4 is generated via running: a) `REAL_WORLD_SEM.py` b) `simulation_real_world.py` c) `simulation_real_world_recovered.py` d) `get_results_real_world.py` e)`Fig4_edge_intersection_comparison_colorscheme.py`. (Refer to `Fig4_edge_intersection_comparison.ipynb` was used to generate the colorful version of the plots and `Fig4_edge_intersection_comparison_colorscheme.ipynb` was actually used to generate the plot with monochromatic color schemes.) Note very importantly, `Fig4_edge_intersection_comparison.py` will produce error message for partial datasets that we have not run the `simulation_real_world_recovered.py` for.
5. Figure 5 and Table S1 (with recovered parameters) is generated via running: a) `REAL_WORLD_SEM.py`, b) `Fig5-ScatterPlot-Table1and2.py`. Use `Fig5-ScatterPlot-Table1and2.ipynb` to view the printed table and figure. 
6. In order to generate AUC scores for all of the XGI datasets/bio-medical datasets, please run `LP-REAL-WORLD.py`(note the comments for the difference of generation of real-world xgi datasets and the bio-medical datasets). After the run completed for both temporal dependent datasets and non-temporal datasets, run `Fig5-ScatterPlot-Table1and2.py` for printing of the link prediction results.

</div>

</div>

<h2 align="center">Example Usage </h2>

See `stochastic-em-demo_TOPK.ipynb` for how to generate synthetic hypergraph and get the recovered parameters. Sepcific comments in the file. 

</div>
