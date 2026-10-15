# 3d-embroidery

This repository contains the code of "Fabricating 3D Shapes by Shrinking-Thread Embroidery" by Annika Oehri, Aviv Segall, Amir Vaxman and Olga Sorkine-Hornung published at Siggraph Asia 2026.
Note: This is research code released to support the reproducibility of our results as well as aiding comparisons. It is not a polished or maintained software package, so the code has very limited documentation and has not been optimized for readability, efficiency, or robustness to e.g. wrong inputs. That said, we hope it’s useful for reproducing our results and for comparisons with your own methods.

## Installation

The inverse design code can be built via the standard cmake routine from the folder inverse_design
```
mkdir build
cd build
cmake ..
make
```
and then executed by e.g.
```
./inv_design ../../data/scaled_beetle.obj 
```

## Usage
The inverse design process can be done by essentially clicking through the interface starting from the top. Specifically,
- click on skeletonization to initialize the flattening
- loop operations starts the iterative field optimization process. The default parameters for this can be adjusted beforehand via the GUI right above.
- once converged, one can click on do tracing, which starts an external streamline tracer following the computed vector field. In the streamline tracing GUI, under Streamline options, adjust the parameters such that you have a very dense network of streamlines to later sample (depending on the resolution and size of your mesh, you might want to increase the max values as well as lower the seed distance. For the scaled meshes in the data folder, a seed distance of 0.2 is often reasonable). Click Trace many streamlines to do the tracing, this can be repeated until you're satisfied with the parameters chosen. Make sure to press save streamlines as soon as you are satisfied before closing this GUI, the results will then be read by the inverse design code.
- press postprocess to sample curves from this, again the standard parameters can be adjusted. With this, you're already done! pattern.svg is automatically saved in the res folder (the other files there are not relevant for standard usage), which you can import into e.g. InkStitch in order to get your embroidery pattern and start the fabrication process.


