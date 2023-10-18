# Detection of Diseases in Tomato Leaves

I have developed a deep learning model for the detection of diseases in tomato leaves. This model is a Convolutional Neural Network (CNN) implemented using the TensorFlow framework. The training process was accelerated by utilizing a GPU, which significantly improved the training speed.
<br>
There is a total of 10,000 images distributed across 10 classes. Among these, 8,000 images are allocated for training, 2,000 for validation, and an additional 100 are reserved for testing. 
The classes are named as follows:
1. Tomato___Bacterial_spot.
2. Tomato___Early_blight.
3. Tomato___Late_blight.
4. Tomato___Leaf_Mold.
5. Tomato___Septoria_leaf_spot.
6. Tomato___Spider_mites_Two-spotted_spider_mite.
7. Tomato___Target_Spot.
8. Tomato___Tomato_Yellow_Leaf_Curl_Virus.
9. Tomato___Tomato_mosaic_virus.
10. Tomato___healthy.
 
The dataset can be downloaded from Kaggle using this link: https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf<br>
The achieved accuracy score stands at an impressive 94%, while the loss metric recorded a value of 1.00.
<br>
##################################################################

The model summary is as follows:
<br>
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 conv2d (Conv2D)             (None, 222, 222, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 111, 111, 32)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 109, 109, 64)      18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 54, 54, 64)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 52, 52, 64)        36928     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 26, 26, 64)       0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 24, 24, 64)        36928     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 12, 12, 64)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 10, 10, 64)        36928     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 3, 3, 64)          36928     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 1, 1, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 64)                0         
                                                                 
 dense (Dense)               (None, 64)                4160      
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 171,914<br>
Trainable params: 171,914<br>
Non-trainable params: 0<br>
_________________________________________________________________
None

