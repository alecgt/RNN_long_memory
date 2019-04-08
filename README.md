# A Statistical Investigation of Long Memory in Language and Music

This repository contains code, data, and illustrative Juypiter notebook detailing the analysis of natural language and music data, along with RNN models trained on such data, using statistical tools developed for long memory stochastic processes. Full details of this methodology and the associated experimental results can be found in the paper:

Alec Greaves-Tunnell and Zaid Harchaoui<br/>
A Statistical Investigation of Long Memory in Language and Music<br/>

## Structure of the repository

The main concepts and experimental results of the paper are illustrated in the Jupyter notebook `Long_Memory_in_Language_and_Music.ipynb`; we recommend any interested users to start here for an introduction to the statistical long memory analysis of RNNs. The companion notebook `Data_Downloads_and_Embeddings.ipynb` offers further information and additional code for users interested in the details of the experiments reported in the paper.

## Dependencies and setup

Code for this notebook is implemented in Python 3. The major dependencies are on 

- [NumPy](http://www.numpy.org/) 
- [SciPy](https://www.scipy.org/)
- [PyTorch](https://pytorch.org/) (Code was implemented in `PyTorch 0.4` but is compatible with version `1.0`)
- [matplotlib](https://matplotlib.org/)

We recommend using [Anaconda](https://www.anaconda.com/distribution/) to create a virtual environment in which this code can be run. With Anaconda installed, a user can clone this repo and set up an environment with the required dependencies via:

```bash
git clone https://github.com/alecgt/RNN_long_memory.git
cd RNN_long_memory/
conda env create -f RNN_long_memory.yml
```

_Note_: Access to GPU resources is neither assumed nor required to run the examples in `Long_Memory_in_Language_and_Music.ipynb` or to run the long memory evaluation tools in `src/eval/longmem_estimation.py`. However, these are highly recommended for users wishing to train their own RNNs.
 
## Citation
The code in this repo is available under GPLv3.

If you find these tools useful in your own work, please cite:

```
@article{greaves2019longmem,
  title={A Statistical Investigation of Long Memory in Language and Music},
  author={Greaves-Tunnell, Alexander and Harchaoui, Zaid},
  journal={arxiv},
  year={2019},
}
```