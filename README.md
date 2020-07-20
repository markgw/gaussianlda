# Gaussian LDA
Another implementation of the paper 
[Gaussian LDA for Topic Models with Word Embeddings](http://rajarshd.github.io/papers/acl2015.pdf).

This is a Python implementation based as closely as possible on 
the [Java implementation](https://github.com/rajarshd/Gaussian_LDA) 
released by the paper's authors.

I made a couple of mathematical changes to the training, which I believe 
to be corrections to the Java implementation. However, if you want to 
replicate exactly what the Java implementation does, set `replicate_das=True`
when instantiating the trainer.


## Installation

You'll first need to install the ``choldate`` package, [following its installation 
instructions](https://github.com/modusdatascience/choldate). (It's not 
possible to include this as a dependency for the PyPi package.)

Then install gaussianlda using Pip:
```
pip install gaussianlda
```

## Other packages

Java implementation:
 * [The original Java implementation by the authors of 
   Gaussian LDA](https://github.com/rajarshd/Gaussian_LDA).

Other Python implementations:
 * [Another Python port of the Java implementation](https://github.com/mansweet/Gaussian-LDA-word2vec). 
   Unlike this package, does not include alias sampling or 
   Cholesky decomposition. 
 * [An extension to train using multilingual 
   embeddings](https://github.com/EliasKB/Multilingual-Gaussian-Latent-Dirichlet-Allocation-MGLDA)


## Usage

The package provides two classes for training Gaussian LDA:
 * Cholesky only, `gaussianlda.GaussianLDATrainer`: Simple Gibbs sampler 
   with optional Cholesky decomposition trick.
 * Cholesky+aliasing, `gaussianlda.GaussianLDAAliasTrainer`: 
   Cholesky decomposition (not optional) and the Vose aliasing trick.

The trainer is prepared by instantiating the training class:
 * *corpus*: List of documents, where each document is a list of int IDs 
   of words. These are IDs into the vocabulary and the embeddings matrix.
 * *vocab_embeddings*: (V, D) Numpy array, where V is the number of words 
   in the vocabulary and D is the dimensionality of the embeddings.
 * *vocab*: Vocabulary, given as a list of words, whose position corresponds 
   to the indices using in the data. This is not strictly needed for training, 
   but is used to output topics.
 * *num_tables*: Number of topics to learn.
 * *alpha*, *kappa*: Hyperparameters to the doc-topic Dirichlet and 
   the inverse Wishart prior
 * *save_path*: Path to write the model out to after each iteration.
 * *mh_steps* (aliasing only): Number of Montecarlo-Hastings steps for 
   each topic sample.
 * *replicate_das*: If True, follow exactly the sampling calculations made by the 
   original Java implementation, without trying to correct the calculation 
   of sampling probabilities and MH acceptance ratio.

Then you set the sampler running for a specified number of iterations 
over the training data by calling `trainer.sample(num_iters)`.

## Example

```python
import numpy as np
from gaussianlda import GaussianLDAAliasTrainer

# A small vocabulary as a list of words
vocab = "money business bank finance sheep cow goat pig".split()
# A random embedding for each word
# Really, you'd want to load something more useful!
embeddings = np.random.sample((8, 100), dtype=np.float32)
corpus = [
    [0, 2, 1, 1, 3, 0, 6, 1],
    [3, 1, 1, 3, 7, 0, 1, 2],
    [7, 5, 4, 7, 7, 4, 6],
    [5, 6, 1, 7, 7, 5, 6, 4],
]
output_dir = "saved_model"
# Prepare a trainer
trainer = GaussianLDAAliasTrainer(
    corpus, embeddings, vocab, 2, 0.1, 0.1, save_path=output_dir
)
# Set training running
trainer.sample(10)
```

You can subsequently load the model from the directory it was saved to 
and get to its distributions and parameters, as well as performing inference 
on new documents.
```python
from gaussianlda.model import GaussianLDA

model = GaussianLDA.load(output_dir)
```