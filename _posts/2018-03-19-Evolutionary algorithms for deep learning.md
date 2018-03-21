---
layout: post
title:  Evolutionary algorithms for deep learning
category: Projects
description: Evolutionary algorithms for deep learning
---

# Evolutionary algorithms for deep learning

In recent times, there has been tremendous achievements in the field of deep learning. I certainly do not need to provide a litany of recent events where deep learning has taken the world by storm. If you are reading this out of interest, I would very likely be preaching to the choir. Let us then delve directly into the matter at hand. 

Let us rather look at ways of implementing evolutionary strategies to augment deep learning. This will be a series of posts that will look into ways of improving deep learning algorithms using evolutionary strategies. 

Importantly, this is going to look into realistic scenarios. We are going to generate a (relatively simple) deep learning architecture in [TensorFlow](https://www.tensorflow.org) and optimize them using some evolutionary strategies. 

## 1. The What ?

So what problem are we looking at specifically?

### 1.1. The Mathematical Representation of a Neural Network 

Neural networks approximations of functions. Like the ones you have already encountered in math: \\(  f( \mathbf x ) = \mathbf y\\). \\( \mathbf x \\) goes into the function \\( f() \\), and out comes \\(  \mathbf y \\). Both \\(  \mathbf x \\) and \\(  \mathbf y \\) can be anything. Scalars, vectors, complex vectors, tensors, binary digits, and what have you. \\(  \mathbf y \\) may similarly be eqaually varied. 

For the purposes of this and subsequent articles, we shall assume that both \\(  \mathbf x \\) and \\(  \mathbf y \\) are real valued vectors, not necessarily of the same dimensions. 

Neural networks *approximate* the function \\( f(  \mathbf x) \\) by using \\(N\\) internal parameters called weights (and sometimes biases, for the case of simplicity, we shall call them all weights). We shall represent them with \\( \mathbf W = [\mathbf W_0, \mathbf W_1, \ldots \mathbf W_{N-1} ]  \\). Practically, all forms of neural networks are represented by these weights and their corresponding *connections* (here, I use the word connections loosely). Connections and weights together form the architecture. 

If we represent the neural network with the function 

$$ f_{ \mathbf {W} } ( \mathbf  x) $$

We want to find optimal weights \\(  \mathbf {W_{Opt}}  \\) such that

$$ f_{ \mathbf {W_{Opt}} } ( \mathbf  x) \approx f ( \mathbf  x)  = \mathbf y $$

for all \\( \mathbf x \\).

### 1.2. Learning

Learning is the process of finding a good set of weights \\(  \mathbf {W_{Opt}} \\). Typical ways of doing this is by some form of gradient descent. Typical algorithms used are the [Steepest Descent](http://mathworld.wolfram.com/MethodofSteepestDescent.html), [Adagrad](https://cs.stanford.edu/~ppasupat/a9online/1107.html), [RMSprop](https://www.coursera.org/learn/deep-neural-network/lecture/BhJlm/rmsprop), and [Adam](https://arxiv.org/abs/1412.6980). 

## 2. The Why ?

There are several problems associated with gradient-based approaches, especially when the optimization dimensionality increases. 

1. The error function *must* be differentiable
2. The error surface *must* be convex (and relatively noiseless)
3. The path toward descent is one along a local minima

Some of these problems can be averted using other non-gradient techniques. These include forms of Evolutionary Algorithms (such as [Genetic Algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm), [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution), etc.), or methods such as [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) or  [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing). 

In this article we shall look at one particular algorithm - Genetic Algorithms for the optimization process. In a later article, we shall look at how both the Genetic Algorithm and the a Adam can work together to produce better results. 

## 3. The How ... Show me the code already !!!

The code is available in the repo: [NNoptExpt](https://github.com/sankhaMukherjee/NNoptExpt). Constructive criticism is always welcome.

Note that although the code shown below is written using the TensorFlow library, this framework can easily be used for any type of class used.   The code shown below is significantly simplified and all error checking, docstrings, logging removed so that it is much easier to follow.

### 3.1. A Generic Neural Network Class

So lets create a simple neural network class [full code here](https://github.com/sankhaMukherjee/NNoptExpt/blob/master/src/lib/NNlib/NNmodel.py).  This is relatively simple code. The neural network that we have created is a simple sequential model. The input to this is simply the input (`inpSize`) and  output (`opSize`) sizes, the number of neurons in each layer (`layers`) passed in as a list, and their corresponding activations (`activations`). Each activation is one of the allowed TensorFlow [activations](https://www.tensorflow.org/api_guides/python/nn#Activation_Functions), or `None` in which case, no activation is applied.

```python
class NNmodel(): 
    
    def __init__(self, inpSize, opSize, layers, activations):

        self.fitted = False
        self.currentErrors = None

        self.Inp     = tf.placeholder(dtype=tf.float32, shape=inpSize, name='Inp')
        self.Op      = tf.placeholder(dtype=tf.float32, shape=opSize, name='Op')
        
        self.allW, self.allWPH, self.allAssignW = [], [], []
        self.allB, self.allBPH, self.allAssignB = [], [], []

        self.result  = None

        prevSize = inpSize[0]
        for i, l in enumerate(layers):

            tempW       = tf.Variable( 0.1*(np.random.rand(l, prevSize) - 0.5), dtype=tf.float32, name='W_{}'.format(i) )
            tempWPH     = tf.placeholder(dtype=tf.float32, shape=(l, prevSize), name='PHW_{}'.format(i))
            tempAssignW = tf.assign(tempW, tempWPH, name='AssignW_{}'.format(i))

            tempB       = tf.Variable( 0, dtype=tf.float32, name='B_{}'.format(i) )
            tempBPH     = tf.placeholder(dtype=tf.float32, shape=tuple(), name='PHB_{}'.format(i))
            tempAssignB = tf.assign(tempB, tempBPH, name='AssignB_{}'.format(i))

            self.allW.append( tempW )
            self.allWPH.append( tempWPH )
            self.allAssignW.append( tempAssignW )

            self.allB.append( tempB )
            self.allBPH.append( tempBPH )
            self.allAssignB.append( tempAssignB )

            if i == 0:
                self.result = tf.matmul( tempW, self.Inp ) + tempB
            else:
                self.result = tf.matmul( tempW, self.result ) + tempB

            prevSize = l

            if activations[i] is not None:
                self.result = activations[i]( self.result )

        self.err = tf.sqrt(tf.reduce_mean((self.Op - self.result)**2))
        self.init = tf.global_variables_initializer()

        return
        
    # more code here
```

#### 3.1.1. Special Aside: Inserting external values into the weights

If you want to insert values into a TensorFlow `Variable`, you will need to use the [`tf.assign`](https://www.tensorflow.org/api_docs/python/tf/assign) operation. This, like all other TensorFlow operation will *extend* the tensor graph every time it is called within a session. Hence, we need to generate a separate `tf.assign` operator for *every* weight we want to be able to change within the graph, and associate a placeholder for *each of these assign operations*. This is what you see in lines 23, 24, 27, 28 of the code. 

Similarly, you only want a single `tf.global_variables_initializer()` operation. We save this within the initializer as well. 

#### 3.1.2. Other Helper Methods

Let us generate a set of helper classes that will be instrumental in making this amenable for optimization using evolutionary techniques.

##### 3.1.2.1. Getting and Setting Weights

These functions will allow us to get the current weights, and set the weights of the neural network. These functions, when implemented, will allow other algorithms to change the weights of the network within a TensorFlow session. The other algorithms no longer need to remember the details of the neural network. This will allow for arbitrarily complex neural networks to be optimized. 

```python
def getWeights(self, sess=None):
    weights = None
    if sess is not None:
        weights = sess.run( self.allW + self.allB  )
        return weights

    with tf.Session() as sess:
        sess.run(self.init)
        weights = sess.run( self.allW + self.allB )
        return weights

def setWeights(self, weights, sess):
    Nw = len(self.allW)
    Ws = weights[:Nw]
    Bs = weights[Nw:]

    for i, (w, mW) in enumerate(zip(Ws, self.allAssignW)):
        sess.run(mW, feed_dict={ 'PHW_{}:0'.format(i) : w } )

    for i, (b, mB) in enumerate(zip(Bs, self.allAssignB)):
        sess.run(mB, feed_dict={ 'PHB_{}:0'.format(i) : b } )
```

##### 3.1.2.2. Calculating Errors and Making Predictions

Getting error values and making predictions. Again, these are self-explanatory. This reduces the burden on the optimizer in knowing about the details of the neural network. The last convenience functions is not necessary, but good to have. Given a set of weights, this will return all errors associated with them. Its utility will become quickly apparent when we generate populations of weights.

```python
def errorValW(self, X, y, weights):
    with tf.Session() as sess:
        sess.run(self.init)
        self.setWeights(weights, sess)
        errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})

    return errVal

def predict(self, X, weights):
    with tf.Session() as sess:
        sess.run(self.init)
        self.setWeights(weights, sess)
        yHat = sess.run(self.result, feed_dict = {self.Inp: X})

    return yHat
    
def errorValWs(self, X, y, weightsList):
        errVals = []
        with tf.Session() as sess:
            sess.run(self.init)
            for weights in weightsList:
                self.setWeights(weights, sess)
                errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                errVals.append(errVal)

        return errVals
``` 


### 3.2. Initialization Example

How do we initialize this? The code below should be sufficient as an explanation. 

 - The input is 2-dimensional
 - There are 2 hidden units (with 5 and 8 neurons respectively), and 
 - The output is single dimensional. 
 - Activations for the two hidden layers are `tf.tanh`, while the last layer isn't activated (useful for regression problems). 

```python
inpSize     = (2, None)
opSize      = (1, None)
layers      = (5, 8, 1)
activations = [tf.tanh, tf.tanh, None]
model1      = NNmodel(inpSize, opSize, layers, activations)
```

### 3.3. Using this NN class

Now we can quickly check whether the weights may be properly transferred across multiple sessions. Weights of the NN are obtained in one session, 5 added to the original weights, and these values inserted into the NN in a new session. As can be seen, the new session contain brand new weights.  

```python
with tf.Session() as sess:
    sess.run(model1.init)
    weights = model1.getWeights()
    print(weights)
    weights = [w+5 for w in weights]
    
with tf.Session() as sess:
    sess.run(model1.init)
    model1.setWeights(weights, sess)
    weights1 = model1.getWeights(sess)
    
print(weights1)
```
### 3.4. The GA Class

## 4. Results

## 5. Conclusion


