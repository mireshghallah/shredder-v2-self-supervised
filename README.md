# Shredder-v2-self-supervised


Code to Shredder: Learning Noise Distributions to Protect Inference Privacy, version2, using self-supervision (https://arxiv.org/abs/1905.118140). By FatemehSadat Mireshghallah (fmireshg@eng.ucsd.edu)

In this repository you can find the code to shredder, and also the .npy files created through sampling, so you do not need to run everything from scratch, you can use the pre-existing ones.

# step by step guide:
1. To do noise training, and save trained samples, run "train-LeNet-param-controlled-self-super-std20-ep27.py". This is a script that generates two .npy files with multiplicative and additive noise tensors. Since this is a one time thing and takes a while, we have provided this named "self-super-std20-nonsen-27ep-noise-2.npy" and "self-super-std20-nonsen-27ep-weight-2.npy" which are the additive and multiplicative noises, respectively.

Note1: The noise tensors are initialized using laplace distribution, and the training is monitored using SNR (we have used SNR as a proxy for mutual information, during training, the lower the SNR, the lower the mutual information and the higher the privacy).

Note2: The noise added here is very aggressive, which means it has an extremely low SNR (0.0002), so it takes for the network a while to adjust. You can test smaller noise (by changing the initialization scale from 20 to 3 for example), and have a faster convergence. Also, during the first epochs, the network is trying to create the clustered representations, so you do not see any improvement in the accuracy. 

2. To sample from the trained noise and save activations for calculating the mutual information, run sample-for-mutual-info-alexnet.py. The results of this step are also provided, with the names: noisy-activation-4-laplace-MI.npy, original-activation-4-laplace-MI.npy,  and original-image-4-laplace-MI.npy which is over 100 mb (around 600mb) and we had to upload it to "https://ufile.io/jhz2d8r7"

3. To see the Mutual Info, you should first have the ITE toolbox cloned (https://bitbucket.org/szzoli/ite-in-python/src/default/). Then, run notebook "mutual_info_ITE-laplace-04.ipynb"


Please do not hesitate to contact me in case of any issues

# Citation
If you used the code or the material in the paper, please cite us suing:


@article{DBLP:journals/corr/abs-1905-11814,
  author    = {Fatemehsadat Mireshghallah and
               Mohammadkazem Taram and
               Prakash Ramrakhyani and
               Dean M. Tullsen and
               Hadi Esmaeilzadeh},
  title     = {Shredder: Learning Noise to Protect Privacy with Partial {DNN} Inference
               on the Edge},
  journal   = {CoRR},
  volume    = {abs/1905.11814},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.11814},
  archivePrefix = {arXiv},
  eprint    = {1905.11814},
  timestamp = {Mon, 03 Jun 2019 13:42:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-11814},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

# License

This software is Copyright © 2019 The Regents of the University of California. All Rights Reserved. Permission to copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs appear in all copies. Permission to make commercial use of this software may be obtained by contacting:

Office of Innovation and Commercialization

9500 Gilman Drive, Mail Code 0910

University of California

La Jolla, CA 92093-0910

(858) 534-5815

invent@ucsd.edu

This software program and documentation are copyrighted by The Regents of the University of California. The software program and documentation are supplied “as is”, without any accompanying services from The Regents. The Regents does not warrant that the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
