## Our 14 model tested with my dataset

* A Vast Dataset for Kurdish Digits and Isolated Characters Recognition

    * link of the dataset in data mandaly [here](https://data.mendeley.com/datasets/zb66pp7vjh)

* A vast dataset for Kurdish handwritten digits and isolated characters recognition from sciencedirect journal

    * Link of the paper about my dataset [here](https://www.sciencedirect.com/science/article/pii/S2352340923001324)
 
* My best model in our model cnn that we tested above is Alex Net Modification 

|**Model Name**|**Number Of Feature**|**Number Of Layer**|**optimization**|**Batch Size**|**Epoch Number**|**Training Accuracy**|**Training Loss**|**Validation Accuracy**|**Validation Loss**|
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
|[CNN 2X GPU](1/Character_cnn_2x_gpu_digit_recognization_(0.99).ipynb)|1,029,283|20 Layers|Adam|10000|100|0\.9908|0\.0285|0\.9447|0\.2521|
|[CNN 1 GPU](2/cnn_0_99_.ipynb)|3,300,259|8 Layers|adam|32|30|0\.9941|0\.0463|0\.9602|0\.8927|
|[CNN 1 GPU](3-4/cnn_0_99_good.ipynb)|196,387|17 Layers|adam|32|100|0\.9712|0\.1050|0\.9808|0\.0717|
|[Timm](5/transfer_learning_with_timm_models_and_pytorch_0_97_Best.ipynb)|5,288,548|4 Layers|rmsprop|1024|40|0\.9739|0\.0108|0\.9739|0\. 0108|
|[CNN 1 GPU](6/lenet_5_model_with_99_accuracy.ipynb)|63,831|8 Layers|adam|32|350|0\.9695 |0\.1081 |0\.9380 |0\.1959|
|[CNN 1 GPU](7/cnn_using_keras_98.ipynb)|371,299|8 Layers|adam|32|350|0\.9990|0\.0065|0\.9703|0\.7782|
|[CNN 1 GPU](8/cnn_99.ipynb)|` `161,027|18 Layers|adam|128|200|<p>0\.9663</p><p></p>|<p>0\.0661</p><p></p>|<p>0\.9832</p><p></p>|<p>0\.0653</p><p></p>|
|[CNN 1 GPU](9/pytorch-1-0-1-on-mnist-acc-99-8.ipynb)|<p>840,595</p><p></p>|18 Layers|Adam|16|50|<p>0\.9302</p><p></p>|<p>0\.2112</p><p></p>|<p>0\.9325 </p><p></p>|<p>0\.1853 </p><p></p>|
|[CNN 1 GPU](10/96-with-pytorch-resnet%20(1).ipynb)|68,106|6 Layers|adam|100|20|0\.9986|0\.0135|0\.0259|15\.2285|
|[Resnet18](11/cnn-resnet-from-scratch-top-10-0-1-2-3-4.ipynb)|11,689,512|3 Layers|SGD|64|10|<p>94\.0888</p><p></p>|<p>0\.182</p><p></p>|<p>96\.3066</p><p></p>|<p>0\.028</p><p></p>|
|[Resnet19](12/cnn.ipynb)|52,094,307|27 Layers|adam|300|30|<p>0\.9719</p><p></p>|<p>0\.1059</p><p></p>|<p>0\.9743 </p><p></p>|<p>0\.1255</p><p></p>|
|[CNN 1 GPU](13/basic-number-classifier.ipynb)|3,300,259|8 Layers|adam|32|13|0\.9998|0\.05789|0\.9998|Very Bad|
|[Alex Net Modification](14/Version_1_Final_Model.ipynb)|82,300,195|33 Layers|<p>rmsprop</p><p></p>|200|200|0\.9999|0\.00024|0\.9900|0\.0923 |


## Related Work

| Reference | Research Focus | Methodology | Results |
| --- | --- | --- | --- |
| Bayan Omar Mohammed | Kurdish handwriting character recognition | Extracted geometric moment features for shape characters and geometric moment features. Utilized Invariant Discretization. | Improved recognition level for solitary handwritten Kurdish letters. |
| Zebardast et al. | Kurdish language (Latin scripts) identification using artificial neural networks | Employed Multilayer Perceptron (MLP) and backpropagation learning algorithm. | Achieved a performance of 81.2677% during the evaluation stage and 85.1535% during the training stage. |
| Zarro and Anwer | Hybrid Hidden Markov Model (HMM) and Harmony Search algorithm for character recognition | Used HMM model for dividing characters into manageable groups and a common directional matrix. Implemented Harmony Search algorithm with fitness function measurements. | Attained a successful recognition rate of 93.52%. |
| Yaseen and Hassani | OCR platform for categorizing and partitioning Kurdish texts written in Persian or Arabic | Developed segmentation techniques based on contour labeling. Conducted tests on different fonts, text sizes, and image resolutions. | Achieved an average recognition rate accuracy of 90.82%. Accuracy was reduced in some cases depending on font style, size, or the presence of multiple fonts. |
| Mohammed et al. | Kurdish Offline Handwritten Text Dataset | Released a dataset with 4304 texts authored by 1076 volunteers. Dataset contained 17466 lines of text. | No information available on the performance or correctness assessment of the dataset at the time of writing. |
| Idrees and Hassani | Recognition of printer characters | Not specified | Not specified |
| Ahmed et al. | Kurdish handwritten character identification using deep learning algorithms | Utilized training dataset for a language with a similar script to address sparse data. | Achieved an accuracy rate of 95.45% for Kurdish handwritten character identification using deep learning algorithms. |


## CNN Model

Convolutional Neural Networks (CNNs) have revolutionized the fields of handwritten character recognition and image classification. These deep learning algorithms have significantly improved the accuracy and efficiency of character recognition systems by automatically learning relevant features directly from raw pixel inputs. Traditional methods relied on manual feature extraction, but CNNs eliminate the need for this step by leveraging their hierarchical structure to learn low-level and high-level features from the input images. This hierarchical feature extraction allows CNNs to capture intricate details in handwritten characters, making them robust and adaptable to different handwriting styles.

Similarly, in image classification, CNNs have achieved remarkable success. They can automatically learn hierarchical representations of visual features, allowing them to recognize complex patterns and objects. By capturing spatial dependencies between pixels in an image, CNNs excel at distinguishing between different objects or classes. The ability to learn these features directly from raw pixel data enables CNNs to achieve high accuracy in image classification tasks, surpassing traditional machine learning algorithms. Techniques like data augmentation, transfer learning, and ensembling further enhance the performance of CNNs in these tasks.

## AlexNet

AlexNet, a powerful deep convolutional neural network (CNN) model, has showcased exceptional performance in handwritten character recognition and image classification. With your implementation of AlexNet on the Central Kurdish digits and isolated characters dataset, promising results have been achieved. AlexNet's hierarchical architecture allows it to learn relevant features from raw pixel inputs, capturing intricate details and variations in handwritten characters. This makes it highly accurate in recognizing and classifying handwritten digits and characters. Additionally, AlexNet's deep architecture enables it to recognize complex patterns and objects in image classification tasks. Its capabilities make it an invaluable tool for computer vision applications, showcasing the potential for further advancements in handwritten recognition and image classification.

* My AlexNet model used in my research can by found [here]()