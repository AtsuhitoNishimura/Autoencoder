# Autoencoder_Viewer
 ”Autoencoder_Viewer" is a tutorial on deep learning using Python and Tensorflow.
 
# DEMO
https://user-images.githubusercontent.com/66617189/187677477-a23d1d89-cc08-4ada-a9fc-bcb101715884.mp4
* **Loss graph**  
    Draws training data and test data loss by epoch.
    ![Autoencoder_Viewer_Loss_Graph](https://user-images.githubusercontent.com/66617189/187687722-105c2761-ec1d-46e5-9e27-b4f190e69624.png)

* **Inference result**  
    The original image is displayed on the upper side and the result of inference is displayed on the lower side.    
![Autoencoder_Viewer_Inference_Result](https://user-images.githubusercontent.com/66617189/187684810-e297a00e-d98d-4b2b-9994-a5dbd50ee8b3.png)

 
 
# Features
Autoencorder_Viewer used [Tensorflow](https://www.tensorflow.org/).
 
```python
import Tensorflow
```
 
# Requirement
* Python 3.10.4
* Tensorflow 2.9.1
 
Environments under [Anaconda for Windows](https://www.anaconda.com/distribution/) is tested.

```bash
conda create -n [env_name] python=3
conda activate [env_name]
```
# Overview
This project consists of two programs.

* **main.py**  
    Specify command line arguments and call model.py.
* **model.py**  
    After the model is defined, data is loaded, trained, and the results are displayed.

# Usage
After Install the necessary libraries with pip command and run "main.py".

```bash
git clone https://github.com/AtsuhitoNishimura/Autoencorder.git
cd Autoencoder
pip install -r requirements.txt
python main.py
```

In main.py, four command line arguments can be specified

| Option | Description | Default |
| :---:  | :---:  | :---:  |
| --train_path, -train | Train data path | "data/train" |
| --test_path, -test | Test data path | "data/test" |
| --epochs, -e | Number of epochs | 150 |
| --bach_size, -b | Size of each image batch | 8 |

# Note
I don't test environments under Linux and Mac.

# Author
* AtsuhitoNishimura
 
# License
"Autoencoder_Viewer" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
 
Have fun studying deep learning!
 
Thank you!
