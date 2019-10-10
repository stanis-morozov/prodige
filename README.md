# PRODIGE: Probabilistic Differentiable Graph Embeddings
A supplementary code for Beyond Vector Spaces: Compact Data Representation as Differentiable Weighted Graphs.
https://arxiv.org/pdf/1910.03524.pdf

# What does it do?
It learns weighted graph representation for your data end-to-end by backpropagation.
<img src="https://raw.githubusercontent.com/neurips-anonymous/prodige/master/demo.png" width=600px>

# What do i need to run it?
* Get as many CPUs as you can
  * We do not support GPU pathfinding (yet)
* Use any popular 64-bit Linux operating system
  * Tested on Ubuntu16.04, should work fine on most linux x64 and even MacOS;
  * On other operating systems we recommend using Docker, e.g. [pytorch-docker](https://hub.docker.com/r/pytorch/pytorch)
* Install the libraries required to compile C++ parts of PRODIGE
  * ```sudo apt-get install gcc g++ libstdc++6 wget curl unzip git```
  * ```sudo apt-get install swig3.0``` (or just swig)
  

# How do I run it?
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install packages from `requirements.txt`
 * It is critical that you use __torch >= 1.1__, not 1.0 or earlier 
 * You will also need jupyter or some other way to work with .ipynb files
4. Open jupyter notebook in `./notebooks/` and you're done!
