# Copy-Move Detection on Digital Image using Python

## Description
This is an implementation of python script to detect a copy-move manipulation attack on digital image based on Overlapping Blocks.

This script is implemented with a modification of two algoritms publicated in a scientific journals:
1. Duplication detection algorithm, taken from [Exposing Digital Forgeries by Detecting Duplicated Image Region](http://www.ists.dartmouth.edu/library/102.pdf) (old link is dead, go to [alternative link](https://www.semanticscholar.org/paper/Exposing-Digital-Forgeries-by-Detecting-Duplicated-Popescu-Farid/b888c1b19014fe5663fd47703edbcb1d6e4124ab)); Fast and smooth attack detection algorithm on digital image using [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis), but sensitive to noise and _post region duplication process_ (explained in the paper above)
2. Robust detection algorithm, taken from [Robust Detection of Region-Duplication Forgery in Digital Image](http://ieeexplore.ieee.org/document/1699948/); Slower and having rough result attack detection algorithm but are considered robust towards noise and _post region duplication process_

More information can be found at Springer that you can find it [here](https://link.springer.com/chapter/10.1007%2F978-3-030-73689-7_39)

### How do we modify them?

We know that Duplication detection algorithm (Paper 1) has `coordinate` and `principal_component` features, and then on Robust detection algorithm (Paper 2) it has `coordinate` and `seven_features` mentioned inside the paper.

Knowing that, we then attempt to give a tolerance by adding all of the features like so:

![Modification diagram](/assets/modification_diagram.PNG?raw=true) 

and then sort it lexicoghrapically.

The principal component will bring similar block closer, while the seven features will also bring closer similar block that can't be detected by principal component (that are for example blurred).

By modifying the algorithms like mentioned above, this script will have a tolerance regarding variety of the input image (i.e. the result will be both smooth and robust, with a trade-off in run time)

## Example image
### Original image
![Original image](/assets/dataset_example.png?raw=true) 
### Forgered image
![Forgered image](/assets/dataset_example_blur.png?raw=true)
### Example result after detection
![Result image](/output/20230202_091213_lined_dataset_example_blur.png)

## Getting Started
```python3
  python3 copy_move_detection/detect.py
```
## Acknowledgments
 This assignment is developed from https://github.com/rahmatnazali/pimage.git. Thanks a lot for your help.
 
# Detecting-Copy-Move-Attack-On-Digital-Images
