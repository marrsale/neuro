class Neuron
  attr_accessor :weight
  attr_reader :activation,:predecessors, :successors

  def initialize(opts={})
    # pass block for non-default activation function (takes two arguments, a weight and a network input)
    # initialize weight and references to predecessors/successors
  end

  def net_input=(x)
    @activation = x
  end

  def output
    if self.input?
      return @activation
    else
      # apply the logistic function onto the activation and return it
    end
  end

  def input?
    @is_input_neuron
  end
end

# NOTES
# TODO: LEARNING RATE PARAM?
# TODO: ACTIVATION FUNCTIONS?
# TODO: DERIVATIVES OF ACTIVATION FUNCTIONS?
# the edges hash:
  # edges: {
  #   neuron_j1: { neuron_k1: Xj1_k1,
  #                 ... },
  #   neuron_j2: { neuron_k1: Xj2_k1,
  #                 ... },
  #   ...
  #   neuron_jK: { neuron_k1: XjK_k1,
  #                 ... } # k is the number of neurons in the preceding layer
  # }
#   - the values stored here are decimal numbers used as weights between neuron layers
#   - the hash uses the neuron objects themselves as keys
#   - the edges hash is composed of N many other hashes, where N is the number of hidden layers plus 1 (the output layer)
#     these hashes are in turn accessed to retrieve the weights
#   - the order in which these are accessed is `edges[successor][predecessor]`,
#     this convention helps us to avoid having to assign duplicate hashes for bidirectional reference

##THE ALGORITHM (for a MLP with one hidden layer)
# training_iteration  = 1
# do # ... until the error term (or momentum) is satisfactorily minimized ceased
while @training
  # FEED FORWARD
  # 1. Apply input vectors to input neurons
  input.each.with_index do |neuron, j|
    neuron.net_input = vectorize(training[training_iteration][:input][j])
  end

  # 2. Calculate the net input values to the hidden neurons
  hidden.each do |neuron|
    sum = 0
    neuron.predecessors.each do |pred|
      sum += pred.output * edges[neuron][pred]
    end
    neuron.net_input = neuron.bias_weight + sum
  end

  # 3, 4. Calculate the outputs from the hidden layer neurons
  #       Calculate the net input values for the output layer
  output.each do |neuron|
    sum = 0
    neuron.predecessors.each do |pred|
      sum += pred.output * edges[neuron][pred]
    end
    neuron.net_input = neuron.bias_weight + sum
  end

  # 5. Calculate the output values for the output layer
  res = output.map do |neuron|
    neuron.output
  end

  # BACKPROPAGATE ERRORS
  if @training?
    # 6. Calculate error terms for the output units
    #    Don't apply the changes yet; calculate and set them aside until we calculate errors for whole network
    output_errors = output.map.with_index do |neuron, j|
      # TODO
      result = (training[training_iteration][:output][j] - neuron.output)*(DERIVATIVE_OF_OUTPUT_ACTIVATION_FUNCTION_ON_NETINPUT)
    end

    # 7. Calculate the error terms for the hidden layer neurons
    hidden_errors = hidden.map.with_index do |neuron, j|
      sum = 0
      neuron.successors.each do |succ|
        sum += output_errors[j] * edges[succ][neuron]
      end
      # TODO
      result = (DERIVATIVE_OF_HIDDEN_ACTIVATION_FUNCTION_ON_NETINPUT)*(sum)
    end

    # 8. Update weights on output layer
    output.each.with_index do |neuron, j|
      neuron.predecessors.each do |pred|
        edges[neuron][pred] += (@learning_rate_param)*(output_errors[j])*(pred.output)
      end
    end

    # 9. Update weights on hidden layer
    hidden.each.with_index do |neuron, j|
      neuron.predecessors.each do |pred|
        edges[neuron][pred] += (@learning_rate_param)*(hidden_errors[j])*(pred.output)
      end
    end

    # 10. Calculate the error term; this is the metric for how well the network is learning
    err_sum = 0
    output_errors.each do |error|
      err_sum += (error ** 2)
    end
    @error_term = (err_sum/2)
  end

  # next training row
  training_iteration += 1
end
