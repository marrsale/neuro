require 'neuron'
# Multilayer Perceptron
class MLP
  # NOTES
  # TODO: LEARNING RATE PARAM?
  #       - fix this at one for now, for pure gradient descent
  # TODO: ACTIVATION FUNCTIONS?
  #       - default if one isn't provided
  #       - (1/Math.exp((1.0 + Math.exp(-H * value))))  # sigmoid/logistic: classification
  #       - (some other function here)                  # linear:
  # TODO: DERIVATIVES OF ACTIVATION FUNCTIONS?
  #       - y*(1-y)
  # TODO: GENERIC DERIVATIVE?
  #       - if a derivative isn't provided, generate a new lambda to calculate derivative using the activation function lambda

  # returns a layer of neurons, i.e. an array of Neuron objects with appropriate propagation functions and initialized weights
  def self.generate_layer!(opts={})
    layer = opts[:array]
    case opts[:type]
    if opts[:type] == :input
      opts[:size].times do
        layer << Neuron.new(input: true,
                            successors: opts[:successors])
      end
    else # generating a hidden or output layer
      opts[:size].times do
        layer << Neuron.new(predecessors: opts[:predecessors],
                            successors: opts[:successors])
      end
    end
  end

  # the network's initializer
  # MLP.new(key: value, key: value)
  def initialize(opts={})
    # initialize layers
    @input_size      = opts[:input_size]
    @hidden_size     = opts[:hidden_size] || @input_size
    @output_size     = opts[:output_size] || @input_size
    @layers          = { input: [], hidden: [[]], output: [] }
    self.generate_layer!(array: input, type: :input, successors: @layers[:hidden], size: @input_size)
    self.generate_layer!(array: hidden, predecessors: @layers[:input], successors: @layers[:output], size: @hidden_size)
    self.generate_layer!(array: output, predecessors: @layers[:hidden], size: @output_size)

    # initialize math
    @learning_rate_param             = 0.2
    @propagation_function            = lambda { |x| 1/(1+Math.exp(-1*(x))) }
    @derivative_propagation_function = lambda { |y| y*(1-y) }
  end

  def input
    @layers[:input]
  end

  def hidden(n=0)
    @layers[:hidden][n]
  end

  def output
    @layers[:output]
  end

  ##THE ALGORITHM (for a MLP with one hidden layer)
  # example for training an MLP to learn the XOR function
  # 4000.times do
  #   @mlp.train_pattern(input: [1,1], output: [0])
  #   @mlp.train_pattern(input: [0,1], output: [1])
  #   @mlp.train_pattern(input: [1,0], output: [1])
  #   @mlp.train_pattern(input: [0,0], output: [0])
  # end
  def train_pattern(training_set)
    # FEED FORWARD
    # 1. Apply input vectors to input neurons
    @training_inputs = training_set[:input]
    input.each.with_index do |neuron, j|
      neuron.net_input = @training_inputs[j]
    end

    # 2. Calculate the net input values to the hidden neurons
    hidden.each do |neuron|
      sum = neuron.predecessors.inject(0) do |acc, pred|
        acc += pred.output * neuron.edge(pred)
      end
      neuron.net_input = neuron.bias_weight + sum
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
    result = output.map(&:output)

    # BACKPROPAGATE ERRORS
    # 6. Calculate error terms for the output units
    #    Don't apply the changes yet; calculate and set them aside until we calculate errors for whole network
    output_errors = output.map.with_index do |neuron, j|
      # TODO
      result = (training_set[:output][j] - neuron.output)*(DERIVATIVE_OF_OUTPUT_ACTIVATION_FUNCTION_ON_NETINPUT)
    end

    # 7. Calculate the error terms for the hidden layer neurons
    hidden_errors = hidden.map.with_index do |neuron, j|
      sum = neuron.successors.inject(0) do |acc, succ|
        acc += output_errors[j] * neuron.edge(succ)
      end
      # TODO
      result = (DERIVATIVE_OF_HIDDEN_ACTIVATION_FUNCTION_ON_NETINPUT)*(sum)
    end

    # 8. Update weights on output layer
    output.each.with_index do |neuron, j|
      neuron.predecessors.each do |pred|
        neuron.edge(pred) += (@learning_rate_param)*(output_errors[j])*(pred.output)
      end
    end

    # 9. Update weights on hidden layer
    hidden.each.with_index do |neuron, j|
      neuron.predecessors.each do |pred|
        neuron.edge(pred) += (@learning_rate_param)*(hidden_errors[j])*(pred.output)
      end
    end

    # 10. Calculate the error term; this is the metric for how well the network is learning
    output_errors.inject(0) do |sum, error|
      sum + (error ** 2)
    end
    @last_error_term = (err_sum/2)
  end # def train_pattern()
end # class MLP
