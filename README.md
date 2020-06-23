# Sharp-In

## Improve image quality utility

### Description

The tool is designed to improve the images quality. At the moment, the tool accepts 512x512 JPG RGB images and returns images of the same size. 

The before/after results look as follows:

<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/59.jpg">

<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/75.jpg">

<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/109.jpg">

[See more examples](https://github.com/lacmus-foundation/sharp-in/blob/master/images).

### Usage

1. Clone this project.
2. Download [weights](https://drive.google.com/file/d/1vYSVbBqZt15jGuVkkMWxK98ORmi6yjp-/view?usp=sharing) to the `sharp-in` project folder.
3. Create the `MyImages` folder at the project folder.
4. Put files to predict to the `MyImages` folder and then run `predict.py`. 
5. The resulting images will appear shortly in the `MyImages` folder.

### Your own weights training

The tool was trained to improve photos from drones. If you want to train it for another domain, create your own dataset with the `prepare_dataset_for_superresolution.ipynb` notebook.

#### General idea of the training

You can use any images for training unless they meet the following requirements:
- 512x512 size images as the target data.
- The same images of the reduced quality as the training data.

The dataset preparation notebook works both with annotated and non-annotated images. 

#### Training dataset preparation

From each image in the `dataset_paths` folders it:
1. Takes crop of `initial_crop_size`.
2. Resizes the crop to `crop_size` - this will be **y** images.
3. Compresses it with the `compress_ratios` ratios and reshapes back to `output_size` - this will be **X** images.
4. The resulting **X** and **y** images are stored in the respective folders inside `crops_folder`. 

You can use the notebook with annotated dataset with two types of crops:
- To get crops around target areas, run `get_target_crops()`. 
- To get `crops_per_image` random crops from each source image, run `get_random_crops()`. 

Other parameters:
- `b_files_per_dataset`: qty of files in the folder used for target crops.
- `r_files_per_dataset`: qty of files in the folder used for random crops.

#### Training

When the dataset is prepared, run `train.py` to start the training.

The recommended settings to start with: Adam optimizer with lr=1e-3 and Cosine LR scheduler. The batch size of 8 or 16 is acceptable.



 



