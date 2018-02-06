![csp](http://www.csp-inc.org/wp-content/uploads/2017/04/csp-logo1.png) 
# The Wilderness Society - Linear features identification  

### Author: Tony Chang 
### Institution: Conservation Science Partners
### Project title: Aerial Detection of Human Modification using Convolutional Neural Networks in the BLM Wilderness areas. 
##### Abstract: The goal of this project is to identify road features within BLM areas from aerial based imagery to optimize efforts for determination of human modified regions and untracked land. In this initial scoping, we explore the usage of Convolutional Neural Networks for the task of semantic segmentation. We propose to implement an autoencoder-decoder network architecture for application in road feature identification, with specific targets of unpaved road features.

### Method:
1. Acquire tiled images of remote sensed imagery and matching stylized vector tile.
2. Use paired images as training data for semantic segmentation using a UNet architecture.
3. Optimize training and architecture to reduce cross-entropy loss function for classification of road features segmentation.
4. Initial validation through first order visual analysis.
5. Model convergence metrics of both training and validation dataset, and confusion matrix to assess precision and recall.  
6. Test data accuracy assessment.   

### Introduction:
We implemented an autoencoder-decoder networks as a general-purpose solution to identify unpaved road features in the BLM Tonopah field office region. [Badrinarayanan et al 2015](https://arxiv.org/pdf/1511.00561.pdf) used this genre of machine learning has displayed early success in urban center satellite image translation to OpenStreetMaps style (photo -> map). 

Update: January 2018
After extensive testing with the pix2pix architecture (generative adversarial networks), it was found that there was high difficulty getting the model to converge and problems with 'Mode Collapse' which is common among GANs. This GAN effort was thus abandoned and focus was shifted towards an Autoencoder-Decoder network such as SegNet to provide semantic segmentation of road features. 

**Overview of algorithm design and mechanics:
[pix2pix-method-document](https://affinelayer.com/pix2pix/)**

**Repository of pix2pix model:
[pix2pix-repository](https://github.com/affinelayer/pix2pix-tensorflow)**

**Repository of SegNet model:
[segnet-repository](https://github.com/0bserver07/Keras-SegNet-Basic)**

### Pilot training and testing: 
Early steps were performed to verify if model could be utilized to predict road features on a complex urban landscape containing numerous feature classes that include buildings, green areas, and various road types. This is demonstrated below with a training model using 3000 images over 3084 epochs (training cycle iterations of random sampling with replacement/image augmentation).
 
**Example model training movie on urban landscape: [CSP-pix2pix-training-movie](https://youtu.be/g5tTgevppWw)**

### Results:
Current results are ~~*_mixed_*~~, *good*.

#### Early application results on Northwestern Colorado region:
#### Sample output 1
![pix2pix-sample1](https://lh6.googleusercontent.com/_yOZQl-rQHVGQGgzofkeBRJAlAfzY4Jdfp9epBF70Z7g1GT3nsPX5lYoXZudQ1HLG_9HmdRqwBcfrx-8TPhi=w950-h979-rw)
#### Sample output 2
![pix2pix-sample2](https://lh3.googleusercontent.com/pPIm18UfKnv4eX11h5_ZFUaV4rltEBPJMgiptbdkzzolXqz9OQEs87dl732zmTcKWN45n1sOEFrzcLbbLDwY=w950-h979-rw)


- Difficulty arises in the training due to [mode collapse](http://aiden.nibali.org/blog/2017-01-18-mode-collapse-gans/), a common issue in GAN models [Goodfellow 2017](https://arxiv.org/pdf/1701.00160v3.pdf). 
    - Using ~2m pixel resolution for a 600x600 (1.5km) tile.  
    - Current training dataset 2000 tiles.

- Next iteration for the remainder of the month (November 2017):
    - Use ~1m resolution pixels for a 500x500 (500m) tile.
    - Incorporate more variance in training data (more class color labels).
    - Increase sample size to 10000 tiles. 
    - Predict over the Rock Springs, WY area.

### Next steps:

- ~~Attempt to optimize pix2pix model with more robust training data set of more tiles.~~
- If unable to produce highly accurate classified road maps, use a threshold measure of pix2pix model predicted road pixels to indicate level of *_potential_* human road impact within an aggregated 1km area. After combine with auxiliary data that may include urban sprawl, agriculture, energy development, and transportation layers.  
- Semantic segmentation with Autoencoder-Decoder shows high success rate. Accuracy assessment and model outputs to be posted soon. 
