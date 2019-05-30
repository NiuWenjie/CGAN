it‘s my CGAN and DCGAN code and results. And the performance is evaluated by Pytorch version of  [FID](https://github.com/mseitzer/pytorch-fid).

* Train  a model:

  ```python
  python cgan_mnist.py --save_path mnist_results
  ```
  ``` --save_path ``` should be given.

* Test a model:

  ```python
  python cgan_mnist.py --save_path mnist_results --batch_size 1 
  ```
  If you want to compute FID, you need to set ``` --batch_size 1 ```.
  
  ### CGAN
  | Datasets | FID |
  | :------: | :----: |
  | MNIST |  27.7237|
  | CIFAR-10 | 280.0052 |


  ### DCGAN
  | Datasets | FID |
  | :------: | :----: |
  | MNIST |  |
