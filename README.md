# Machine Learning Madeiteasy

The project is my very first touch experience to Machine Learning. 
Most algorihms are implemented from scratch following Pattern Recognition and Machine Learning textbook by C.M.Bishop 

## Getting Started

Most Machine Learning algorithms in this repository are implemented to attack IRIS classification problem. 
The approaches range from Parametric to Non Parametric methods in density estimation to Bayesian approaches in Generative Probabilistic Models to Neural Networks.
The accuracy rates are shown in table below (see HW3 for Neural Networks):

|                                                                               | No of wrong classification | Performance |
|-------------------------------------------------------------------------------|----------------------------|-------------|
| Maximum likelihood classifier - Gaussian assumption                           | 4/150                      | 97.33%      |
| Maximum likelihood classifier- a mixture of K Gaussians (K=2)                 | 2/150                      | 98.67%      |
| Maximum likelihood classifier - Gaussian KDE                                  | 6/150                      | 96%         |
| K-nearest neighbor                                                            | 6/150                      | 96%         |
| Several 2-class discriminant functions using one-vs-one combination           | 2/150                      | 98.67%      |
| Several 2-class discriminant function using one-vs-rest combination           | 38/150                     | 74.67%      |
| 3-class discriminant function classifier                                      | 28/150                     | 81.33%      |
| 3-class Bayesian classifier using softmax function (generative probabilistic) | 3/150                      | 98%         |

Furthermore, in this repository, several basic probability distribution, sampling methods (Metropolis Hasting, Hamiltonian MC) as well as numerical methods (finding root equations) are implemented for someone to get the ideas behind ML algorithms. 
Latent Semantic Indexing problem are also illustrated with Singular Value Decomposition (SVD).

### Prerequisites

Numpy
Pandas
Matplotlib

### Installing

A step by step series of examples that tell you have to get a development env running

Step-by-step

```
pip install "numpy = your desired version"
```

And Pandas

```
pip install pandas
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* Numpy package- python for numerical analysis
* Pandas package- data manipulation and analysis
* Matplotlib - Used to generate figures and images

## Contributing

Please read [CONTRIBUTING.md]() for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning


## Authors

* **Phuong Hoang Minh* - *Initial work* - [PhuongHoangMinh](https://github.com/PhuongHoangMinh)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License


## Acknowledgments
