# Sharp-In
## Improve image quality utility

### Description

The tool is designed to improve images quality. At the moment it accepts JPG RGB images of 512x512 size and returns images of the same size. 

Results "before - after" look like:

<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/59.jpg">

<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/75.jpg">

<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/109.jpg">

[More examples.](https://github.com/lacmus-foundation/sharp-in/blob/master/images)

### Usage

- clone this project;
- download [weights](https://drive.google.com/file/d/1vYSVbBqZt15jGuVkkMWxK98ORmi6yjp-/view?usp=sharing) to the same folder with **predict.py**;
- create folder **'MyImages'** at the same location and put there files to predict;
- run **predict.py**;
- resulting images will appear shortly in **'MyImages'** folder.


### Train your own weights
Tool was trained to improve photos from drones. If you wish to train it for another domain, you have to create your own dataset with **prepare_dataset_for_superresolution.ipynb** notebook.


#### General idea of training
Any images can be used for training. Tool requires 512x512 shaped images as target data and the same images, but with reduced quality, as training data.
DS preparation notebook works both with annotated and non-annotated images. 


#### Training dataset preparation
From each image in *dataset_paths* folders it:
- takes crop of *initial_crop_size*;
- then resize the crop to *crop_size* - this will be **y** images;
- compress it with *compress_ratios* ratios and reshapes back to *output_size* - this will be **X** images;
- resulting **X** and **y** images being stored in *crops_folder*, in respective folders.

To use notebook with annotated datasets run *get_target_crops()*. In this case crops around target areas will be prepared. 
Otherwise use *get_random_crops()*, it will make *crops_per_image* random crops from each source image.

Other params:
- *r_files_per_dataset*: qty of files in folder use for random crops;
- *b_files_per_dataset*: qty of files in folder use for target crops.


#### Training

When dataset is prepared, training can be started. Use **train.py** for that purpose.

Good settings to start: Adam optimizer with lr=1e-3 and Cosine LR scheduler. Batch size of 8 or 16 is acceptable.



 



