# Neuro

This is an experimental neural network currently suited to creating multilayer perceptrons with a variable number of hidden layers and autoencoders.

Included in the repo is `neuro/examples`, a directory which contains a couple of very basic example programs which can be used to test the ANN: `mlp_example.rb` and `autoencoder_example.rb`. To run the examples, you must first

```bundle install```

to install the gems in the Gemfile, and then the program of choice can be executed with

```bundle exec ruby <file name>```

# Example Programs

## [Multilayer Perceptron (MLP)](http://en.wikipedia.org/wiki/Multilayer_perceptron)

A multilayer perceptron is a mathematical modeling of an animal brain, and can be used in conjunction with training methods (in this case, backpropagation) to train the internal state of the MLP to recognize a function.  The most common use for multilayer perceptrons is for learning to classify data sets that are non-linearly separable, i.e. we can theoretically 'teach' our MLP to recognize classes of things, as long as we can parameterize them meaningfully.

Refer to the `examples/mlp_example.rb` example program included in this repository for an instance of multilayer perceptrons, in which we train our MLP to learn the **Exclusive OR (XOR) function**, or in human terms, we return 1 when (and only when) a single input is 1.  Note that this is a somewhat standard example for demonstrating logistic regression capability, as the XOR function can't be modeled linearly.

In our example program, we first instantiate a new MLP of with two input neurons, two hidden neurons in a single layer, and a single output neuron:
```
mlp = ANN::MLP.new(input: 2, hidden: [2], output: 1)
```
Next, we simply iterate 10,000 times over our training set of only four input sets, training our MLP to recognize each once per iteration:

```
100000.times do |n|
  mlp.train_pattern!(input: [0,0], output: [0])
  mlp.train_pattern!(input: [0,1], output: [1])
  mlp.train_pattern!(input: [1,1], output: [0])
  mlp.train_pattern!(input: [1,0], output: [1])
  # print stuff here...
end
```
Here's some terminal output from a sample run (edited for clarity):
```
For iteration #2000, error term is 0.003656866482768353.
.
.
.
For iteration #98000, error term is 2.2585318513659154e-06.
For input [1, 1]	[0.06164708960224026]
For input [1, 0]	[0.9544334325171219]
For input [0, 0]	[0.019713717703981025]
For input [0, 1]	[0.9532905467326417]
```

Note that, in this case, our gave the desired result with a relatively small margin of error. Due to the randomness of the neurons when they are initialized, performance can be highly variable.

## [Autoencoder](http://en.wikipedia.org/wiki/Autoencoder)

Autoencoders are often used to reduce the dimensionality of an input space by compressing the information.  This is especially useful when used in conjunction with MLPs, to reduce the size of the input vector and thereby limit the amount of noise and processing which has to be done to train an MLP.  Thus, autoencoders are often used as a preprocessing component, with a fully connected MLP at the end for classification upon the compressed data.

For our sample program, we first must generate a set of inputs for our autoencoder to learn.  We don't really care what they are, so we can generate them randomly:

```
input_sets = []
  10.times do
  input_sets << [[0,1].sample, [0,1].sample, [0,1].sample]
end
```

Next, we instantiate our autoencoder:
```
autoencoder = ANN::MLP.new(input: 3, hidden: [2], output: 3)
```

Note that we have 3 input units and 3 output units, but only 2 neurons in the hidden layer: this is how the data compression comes about.  Now, we can train our autoencoder to learn the input set (we train it to return its best approximation of the input):
```
10000.times do |n|
  input_sets.each do |set|
  autoencoder.train_pattern!(input: set, output: set)
  end
  # print stuff...
end
```

Here's some terminal output from an actual run:
```
For iteration #8000, error term is 4.70371785670798e-05.
For input: [0, 0, 0],	Output: [0, 0, 0]
For input: [0, 0, 0],	Output: [0, 0, 0]
For input: [0, 0, 1],	Output: [0, 0, 1]
For input: [1, 1, 1],	Output: [1, 1, 1]
For input: [0, 0, 0],	Output: [0, 0, 0]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 1, 0],	Output: [1, 0, 1]
For input: [0, 0, 1],	Output: [0, 0, 1]
For input: [1, 1, 1],	Output: [1, 1, 1]
For input: [1, 1, 1],	Output: [1, 1, 1]
```

## Stacked Autoencoder

Stacked autoencoders are autoencoders which are created layer-wise, in order to train it in a greedy incremental fashion.  A dimensionality- or noise-reducing autoencoder can be created with few (or one) hidden layers, and new layers can be added later and trained to capture higher-level features.  This allows for more flexibility and dynamism when generating preprocessing units.


Using a stacked autoencoder:
```
autoencoder = ANN::StackedAutoEncoder.new input: 3, hidden: 2

# ... train the first layer

For iteration #8000, error term is 9.124077827543285e-06..
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 1, 1],	Output: [0, 1, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [0, 1, 0],	Output: [1, 1, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 0, 0],	Output: [1, 0, 0]
For input: [0, 0, 1],	Output: [1, 0, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]

# Add another layer
autoencoder.append_layer! 3

# ... train the network with a new layer, now that we've learned some features

For iteration #8000, error term is 1.5303922387867234e-06.
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 1, 1],	Output: [1, 1, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [0, 1, 0],	Output: [0, 1, 0]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 0, 0],	Output: [1, 0, 0]
For input: [0, 0, 1],	Output: [0, 0, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]
For input: [1, 0, 1],	Output: [1, 0, 1]

```

### TODOS
- Incorporate basic tests to prevent regression, bugs (~~MLP~~, Neuron, StackedAutoencoder)
- Refactor serialization of ANN::MLP
- Add option for bias node, including weights to shared neuron
- Implement Graph Network class, for composing graphs of various network types (including serialization)

#### License
This repo and its contents belong to [Alexander Marrs](github.com/marrsale), but can be used, copied, shared and modified by anyone for any ethical purpose as long as attributions to the original author are left in the source.
