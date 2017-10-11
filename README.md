# enlargeme
Code I used for my Master Thesis: Super-resolution of magnetic resonance imaging data using machine learning tools.

## Motivation
Master Thesis as mentioned, but mostly self-edu. Machine Learning became my main field of study in 2015 and since then I used popular frameworks like caffe, matconvnet or torch to build neural networks (mainly for image processing). This time I wanted to implement this structures myself to get more familliar with low level details. Also these are my first steps in unsupervised learning!

## Solution
First version was to create two structures that can extract features from respectively low and high resolution images, and then create non linnear mapping between feature maps. After presenting a LR image to the first structure features would be extracted, then mapped to HR feature maps. Then SR image would be generated. 

For now I am working on training RBM that will have real-valued visible layer (as its input is image) and convolutional connection between layers. When this part is done - after providing CCRBM (Convolutional Continuous RBM) an MRI image, and sampling v->h->v layer output will be at least as good as input image - non linnear mapping will be implemented. The plan is to use a neural network, but I'm not sure about the details. Should it be convolutional as well, or simple MLP will be enough.

## TO DO:
- [x] Implement CCRBM model with activations
- [x] Implement contrastive divergence algorithm
- [x] Add batch version of CD
- [x] Implement Persistant CD algorithm for more efficient learning
- [x] Implement learning with scalar B and C values
- [x] Enable saving models to files and then loading it back to python
- [x] Implement methods for testing current solutions, visualise filters, monitor mean square error
- [x] Compare scalar B and C version with master where there are matrices
- [x] Try to combine both versions by first using scalar values and then matrices B and C (for more efficient learning)
- [x] Implement decent test and train sets in DataHandler, improve normalization and measurements
- [ ] Add logger
- [ ] Implement momentum for contrastive divergence for faster filters training
- [ ] Implement extra improvements mentioned in [2]
- [ ] Write complete test cases to run on remote server
- [ ] Keep README.md up to date
- [ ] Make enalargeme a package that one can install with pip
- [ ] add some CI like Jenkins or Travis
- [ ] suggestions?




#### Usefull information 
[1] Fischer, Igel: An introduction to Restricted Boltzmann Machines (2012)

[2] Hinton: A Practical Guide to Training Restricted Boltzmann Machines (2010)

[3] Norouzi, Ranjbar, Mori: Stacks of Convolutional Restrited Boltzmann Machines for Shift-Invarian Feature Learning

[4] Lee, Largman, Pham, Ng: Unsupervised feature learning for audio calssification using convolutional deep belief networks 

[5] Lee, Grosse, Ranganath, Ng: Convolutional Deep Belief Networks for Scallable Unsupervised LEarning of Hierarchical Representations

[6]Bengio, Delalleau: Justifying and Generalizing Contrastive Divergence (2007)

[7] Chen, Murray: Continuous restricted boltzmann machine with an implementable training algorithm (2003)
