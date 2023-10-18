# Differentiation-of-Atypical-Parkinsonian-Syndromes-Using-Hyperbolic-Few-Shot-Contrastive-Learning
![Overall Framework Flowchart](https://github.com/asd147asd147/Differentiation-of-Atypical-Parkinsonian-Syndromes-Using-Hyperbolic-Few-Shot-Contrastive-Learning/assets/55697983/1761ccb9-600f-4229-890b-b5bf38271015)

1. **Prerequisites**</br>

Basically, deeplearning environment needs to consider lots of things.
Like, verision of cuda, nvidia driver and the Deep learning framework.
So, it is highly recommended to use docker.
I also made my experiment environment by utilizing the docker.
The fundamental environment for this experiment is like below.
> - Ubuntu (Linux OS for using Nvidia docker)
> - pytorch v1.10.0
> - cuda 11.3
> - cudnn 8  

Use Dockerfile to build the same environment. The required python library information is recorded in requirement.txt.

2. **Dataset**</br>
The dataset we used was not disclosed. In order to use the code, the code must be executed by configuring the following folder structure.

>```bash
> Dataset
> | - MSAP
> | - MSAC
> | - PSP
> | - PD
> ㄴ- NC
> ```



3. **Training & Testing**</br>
If you have both data and environment, run the code below to perform learning and testing normally
> ```bash
> python main.py
> ```

Create the weights folder in the top folder to store the weights of the learning model. And if you want to get an embedded image on Poincare Ball, create a temp folder as well.

![UMAP](https://github.com/asd147asd147/Differentiation-of-Atypical-Parkinsonian-Syndromes-Using-Hyperbolic-Few-Shot-Contrastive-Learning/assets/55697983/50760997-e5e7-4b70-a90d-928248574cfb)
