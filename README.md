# A Statistical Investigation of Long Memory in Language and Music

This repository contains code, data, and illustrative Juypiter notebook detailing the analysis of natural language and music data, along with RNN models trained on such data, using statistical tools developed for long memory stochastic processes. Full details of this methodology and the associated experimental results can be found in the paper:

[A Statistical Investigation of Long Memory in Language and Music](http://proceedings.mlr.press/v97/greaves-tunnell19a.html)<br/>
Alec Greaves-Tunnell and Zaid Harchaoui<br/>

## Structure of the repository

The main concepts and experimental results of the paper are illustrated in the Jupyter notebook `Long_Memory_in_Language_and_Music.ipynb`; we recommend any interested users to start here for an introduction to the statistical long memory analysis of RNNs. The companion notebook `Data_Downloads_and_Embeddings.ipynb` offers further information and additional code for users interested in the details of the experiments reported in the paper.

## Dependencies

Code for this notebook is implemented in Python 3. The major dependencies are on 

- [NumPy](http://www.numpy.org/) 
- [SciPy](https://www.scipy.org/)
- [PyTorch](https://pytorch.org/) (Code was implemented in `PyTorch 0.4` but is compatible with version `1.0`)
- [matplotlib](https://matplotlib.org/)


## Setup

We recommend using [Anaconda](https://www.anaconda.com/distribution/) to create a virtual environment in which this code can be run. With Anaconda installed, a user can clone this repo and set up an environment with the required dependencies via:

```bash
git clone https://github.com/alecgt/RNN_long_memory.git
cd RNN_long_memory/
conda env create -f RNN_long_memory.yml
```

_Note_: Access to GPU resources is neither assumed nor required to run the examples in `Long_Memory_in_Language_and_Music.ipynb` or to run the long memory evaluation tools in `src/eval/longmem_estimation.py`. However, these are highly recommended for users wishing to train their own RNNs.

## Data

We have provided some example text and music data that will be required for users interested in running the code in `Long_Memory_in_Language_and_Music.ipynb`. Two simple additional steps will also be required:

1. Unzip the embedded music data:

```bash
cd data/
tar -xvzf bach_cello_suite.tar.gz
```

2. Obtain the GloVe word vectors used to embed the text data. Specifically, we require that the file `glove.6B.50d.txt` be downloaded to the `data/` directory. This is available from Stanford NLP:

```bash
cd data/
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
 
## Citation
If you find these tools useful in your own work, please cite:

```
Greaves-Tunnell, A. & Harchaoui, Z. (2019). A Statistical Investigation of Long Memory in Language
  and Music. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:2394-2403
```


The BibTex reference is:
```
@InProceedings{pmlr-v97-greaves-tunnell19a,
  title = 	 {A Statistical Investigation of Long Memory in Language and Music},
  author = 	 {Greaves-Tunnell, Alexander and Harchaoui, Zaid},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {2394--2403},
  year = 	 {2019},
  volume = 	 {97}
}
```

## Acknowledgments

This work was supported by the Big Data for Genomics and Neuroscience Training Grant 8T32LM012419, NSF TRIPODS Award CCF-1740551, the program ``Learning in Machines and Brains" of CIFAR, and faculty research awards.