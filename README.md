it‘s my CGAN code and results. And the performance is evaluated by Pytorch version of  [FID](https://github.com/mseitzer/pytorch-fid).

| Datasets | FID |
| :------: | :----: |
| MNIST |  27.7237|
| CIFAR-10 | 280.0052 |

* Train  a model:

  ```python
  python cgan_mnist.py --save_path mnist_results
  ```

* Test a model:

  ```python
  python cgan_mnist.py --save_path mnist_results --batch_size 1 
  ```

