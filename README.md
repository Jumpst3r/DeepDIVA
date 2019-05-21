# Introduction

This fork contains modifications to the hisdb runner class. These modifications were made in order to be able to work with our own dataset for
the task of printed and handwritten text identification in mixed documents. To reproduce the U-net experiment described in the paper, install
DeepDiva using the intructions further bellow and execute following command:

``` shell
source activate deepdiva
```

``` shell
python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder ../output_yolo_try2 --dataset-folder datasets/printed-hw-seg/ --ignoregit --model-name unet --epochs 20 --experiment-name unet_ultra_final  --batch-size 16 --crop-size 256 --imgs-in-memory 3 --crops-per-image 500 --optimizer-name Adam --no-val-conf-matrix -j 16 --seed 1451368622
```

This will train, validate and test the model. Note that even though we tell DeepDIVA to train for 20 epochs, the model at epoch 6 will be chosen to run
the tests, as further training overfitts the network.


### DeepDIVA: A Highly-Functional Python Framework for Reproducible Experiments

DeepDIVA is an infrastructure designed to enable quick and intuitive
setup of reproducible experiments with a large range of useful analysis
functionality.
Reproducing scientific results can be a frustrating experience, not only
in document image analysis but in machine learning in general.
Using DeepDIVA a researcher can either reproduce a given experiment with
a very limited amount of information or share their own experiments with
others.
Moreover, the framework offers a large range of functions, such as
boilerplate code, keeping track of experiments, hyper-parameter
optimization, and visualization of data and results.
DeepDIVA is implemented in Python and uses the deep learning framework
[PyTorch](http://pytorch.org/).
It is completely open source and accessible as Web Service through
[DIVAServices](http://divaservices.unifr.ch).

### Additional resources

- [DeepDIVA Homepage](https://diva-dia.github.io/DeepDIVAweb/index.html)
- [Tutorials](https://diva-dia.github.io/DeepDIVAweb/articles.html)
- [Paper on arXiv](https://arxiv.org/abs/1805.00329) 

## Getting started

In order to get the framework up and running it is only necessary to clone the latest version of the repository:

``` shell
git clone https://github.com/DIVA-DIA/DeepDIVA.git
```

Run the script:

``` shell
bash setup_environment.sh
```

Reload your environment variables from `.bashrc` with: `source ~/.bashrc`

## Verifying Everything Works

To verify the correctness of the procecdure you can run a small experiment. Activate the DeepDIVA python environment:

``` shell
source activate deepdiva
```

Download the MNIST dataset:

``` shell
python util/data/get_a_dataset.py mnist --output-folder toy_dataset
```

Train a simple Convolutional Neural Network on the MNIST dataset using the command:

``` shell
python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --no-cuda
```

## Citing us

If you use our software, please cite our paper as (will be updated soon):

``` latex
@inproceedings{albertipondenkandath2018deepdiva,
    archivePrefix = {arXiv},
    arxivId = {1805.00329},
    eprint = {1805.00329},
    author = {Alberti, Michele and Pondenkandath, Vinaychandran and Würsch, Marcel and Ingold, Rolf and Liwicki, Marcus},
    title = {{DeepDIVA: A Highly-Functional Python Framework for Reproducible Experiments}},
    year = {2018},
    month = {apr},
}
```

## License

Our work is on GNU Lesser General Public License v3.0

