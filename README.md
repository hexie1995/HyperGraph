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


</div>

</div>

<h2 align="center">Example Usage </h2>

See `stochastic-em-demo_TOPK.ipynb` for how to generate synthetic hypergraph and get the recovered parameters. Sepcific comments in the file. 

</div>
