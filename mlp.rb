# COPYLEFT ORIGINAL AUTHOR ALEXANDER MARRS (github.com/marrsale / twitter.com/alx_mars)
require './neuron.rb'
# require 'pry'
# Multilayer Perceptron
class MLP
  attr_reader :last_error_term
  # The Multilayer Perceptron initializer
  # MLP::new()
  # ex. for an MLP with am input layer of size 2, hidden layer size 2, output layer size 1, by default logistic/classification propagation
  #   @mlp = MLP.new(input: 2, hidden: [2], output: 1)
  # ex. for an MLP with an input layer n=2, 2 hidden layers n=2, output layer n=1
  #   @mlp = MLP.new(input: 2, hidden: [2, 2], output: 1)
  def initialize(opts={})
    # initialize math
    @learning_rate_param             = 0.20 # TODO: not doing anything with this currently
    @propagation_function            = opts[:propagation_f]     || lambda { |x| 1/(1+Math.exp(-1*(x))) }
    @derivative_propagation_function = opts[:propagation_deriv] || lambda { |y| y*(1-y) }

    # initialize layers
    @input_size  = opts[:input]
    @hidden_size = opts[:hidden] || @input_size
    @output_size = opts[:output] || @input_size
    @num_layers  = opts[:hidden].count
    generate_layers! # initialize all the layers with new neurons and connect them
  end

  def input
    @layers[:input]
  end

  def hidden
    @layers[:hidden]
  end

  def output
    @layers[:output]
  end

  # Method used for evaluating an input vector
  # Returns a vector of outputs from the output layer neurons
  # NOTE: this method is named with a bang because it changes the state of the network
  # ex. for an instance of MLP, @mlp, that has been trained the XOR function:
  #   @mlp.evaluate([1,1]) # => [0]
  #   @mlp.evaulate([0,1]) # => [1]
  def evaluate!(input_set)
    # FEED FORWARD
    # 1. Apply input vectors to input neurons
    @inputs = input_set
    input.each.with_index do |neuron, j|
      neuron.net_input = @inputs[j]
    end

    # 2. Calculate the net input values to the hidden neurons
    hidden.each do |hidden_layer|
      hidden_layer.each do |neuron|
        sum = neuron.predecessors.inject(0) do |acc, pred|
          acc += pred.output * neuron.edge(pred)
        end
        neuron.net_input = neuron.bias_weight + sum
      end
    end

    # 3, 4. Calculate the outputs from the hidden layer neurons
    #       Calculate the net input values for the output layer
    output.each do |neuron|
      sum = neuron.predecessors.inject(0) do |acc, pred|
        acc += pred.output * neuron.edge(pred)
      end
      neuron.net_input = neuron.bias_weight + sum
    end

    # 5. Calculate the output values for the output layer
    @last_result = output.map(&:output)
  end

  ##THE ALGORITHM (for a MLP with one hidden layer)
  # Method used for training the network to a specific set of inputs
  # Returns its error term after every calculation
  # NOTE: this method is named with a bang because it changes the state of the network
  # example for training an MLP to learn the XOR function
  #   4000.times do
  #     @mlp.train_pattern!(input: [1,1], output: [0]) # => Float
  #     @mlp.train_pattern!(input: [0,1], output: [1]) # => Float
  #     @mlp.train_pattern!(input: [1,0], output: [1]) # => Float
  #     @mlp.train_pattern!(input: [0,0], output: [0]) # => Float
  #   end
  def train_pattern!(training_set)
    # First have the network evaluate its input so we can see how well we did
    evaluate!(training_set[:input])

    # BACKPROPAGATE ERRORS
    # 6. Calculate error terms for the output units
    #    Don't apply the changes yet; calculate and set them aside until we calculate errors for whole network
    output_errors = output.map.with_index do |neuron, j|
      # we return an array of two elements, the error value and the neuron it belongs to
      # this is for keying purposes later when we need to update the edges
      [neuron, ((training_set[:output][j] - neuron.output)*(neuron.gradient))]
    end

    # 7. Calculate the error terms for the hidden layer neurons
    hidden_errors = {}
    hidden.each do |hidden_layer|
      hidden_errors[hidden_layer] = hidden_layer.map.with_index do |neuron, j|
        sum = output_errors.inject(0) do |acc, succ|
          acc += succ[1] * neuron.edge(succ[0])
        end
        [neuron, (neuron.gradient)*(sum)]
      end
    end

    # 8. Update weights on output layer
    output.each.with_index do |neuron, j|
      neuron.predecessors.each do |pred|
        neuron.update_edge!(pred, (neuron.edge(pred) + (@learning_rate_param)*(output_errors[j][1])*(pred.output)))
      end
    end

    # 9. Update weights on hidden layer
    hidden.each do |hidden_layer|
      hidden_layer.each.with_index do |neuron, j|
        neuron.predecessors.each do |pred|
          neuron.update_edge!(pred, (neuron.edge(pred) + (@learning_rate_param)*(hidden_errors[hidden_layer][j][1])*(pred.output)))
        end
      end
    end

    # 10. Calculate the error term; this is the metric for how well the network is learning
    err_sum = output_errors.inject(0) do |sum, error|
      sum + (error[1] ** 2)
    end
    @last_error_term = (err_sum/2)
  end # def train_pattern()

  private

  # GENERATE LAYERS ON EACH WITH THEIR APPROPRIATE PREDS/SUCCS
  def generate_layers!
    @layers      = { input: [], hidden: [], output: [] } # note: hidden will be an array of arrays
    @num_layers.times { @layers[:hidden] << [] } # stuff appropriate number of  uninitialized layers into hidden array
    # returns a layer of neurons, i.e. an array of Neuron objects with appropriate propagation functions and initialized weights
    def generate_layer!(opts={})
      layer = opts[:array]
      if opts[:type] == :input
        opts[:size].times do
          layer << Neuron.new(input_node: true,
          successors: opts[:successors])
        end
      else # generating a hidden or output layer
        opts[:size].times do
          layer << Neuron.new(predecessors: opts[:predecessors],
          successors: opts[:successors],
          activation_f: @propagation_function,
          deriv_f: @derivative_propagation_function)
        end
      end
    end
    # Generate the input layer
    generate_layer!(array: input, type: :input, successors: hidden.first, size: @input_size)
    # Generate the hidden layer(s)
    @num_layers.times do |n|
      if n == 0 # first element
        if @num_layers == 1 #first AND last element
          generate_layer!(array: hidden.first, predecessors: input, successors: output, size: @hidden_size[n])
        else # first element, there are more hidden layers to come
          generate_layer!(array: hidden.first, predecessors: input, successors: hidden[n], size: @hidden_size[n])
        end
      elsif n == (@num_layers - 1) # last element, AND not the first element
        generate_layer!(array: hidden[n], predecessors: hidden[n-1], successors: output, size: @hidden_size[n])
      else # neither the first or last hidden layer, keep going
        generate_layer!(array: hidden[n], predecessors: hidden[n-1], successors: hidden[n+1], size: @hidden_size[n])
      end
    end
    # Generate the output layer
    generate_layer!(array: output, predecessors: hidden.last, size: @output_size)
  end
end # class MLP
