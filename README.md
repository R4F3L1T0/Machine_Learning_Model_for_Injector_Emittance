# Machine Learning Model for Injector Emittance
### This repository contains the work I did in collaboration with the _LCLS_ (Linac Coherent Light Source) group at SLAC (Stanford Linear Accelerator Center) as Summer Intern in 2018.

I contribuited to the program to tune the world's first X-ray laser using advanced Machine Learning (ML) techniques.
The people I collaborated with are: Daniel Ratner, Auralee Edelen, Dorian Bohler and many others.

In this repository you will find:

Files
- a detailed Report (**Report_Raffaele_Campanile_SLAC.pdf**) describing each step of the work, from the gathering of the data to the ML model results.

- a, very ugly, Final Presentation (**Final_Talk_Presentation.pptx**) describing the main ideas and the results obtained in this work.  

Folders
- a folder containing the python code (**code**) for the ML model (**neural_network.py**) and for a simple plotter tool (**results_plotter.py**) to compare the ML model performance with the expected results.

- a folder containing the data (**data**), downloaded from the _LCLS_ archive and already prepared to be analyzed.

- a folder called **results** where the performance of the ML model will be saved after you run it both with the TensorBoard callbacks and a copy of the model with its own weights.

- the folder **my_results** contain the best results I managed to get with the given dataset. It's here so you can compare your model with mine. Of course, if manage to get an improvement, feel free to contact me or to add a branch.

The study I performed was a really pioneering one for the _LCLS_ community, since for the first time someone tried to predict emittance values starting from Injector settings.
Given the hardness of the problem we are very happy with the results we got.
