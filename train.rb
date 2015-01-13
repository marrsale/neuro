# Multilayer Perceptron
class MLP
  class Neuron
    attr_accessor :weight
    attr_reader :activation,:predecessors,:successors,:is_input


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
        @activation_function.call(@activation)
      end
    end

    # return the edge between self and a connected neuron
    def edge(neuron)
      # the edge either belongs to self or the other neuron, go get it
      if (edge = @edges[neuron] || neuron.edge(self))
        edge
      else
        raise "Can't return edge.  #{self} isn't connected to #{neuron}"
      end
    end
  end

  # NOTES
  # TODO: LEARNING RATE PARAM?
  #       - fix this at one for now, for pure gradient descent
  # TODO: ACTIVATION FUNCTIONS?
  #       - (1/Math.exp((1.0 + Math.exp(-H * value))))
  # TODO: DERIVATIVES OF ACTIVATION FUNCTIONS?
  #       - y*(1-y)

  # the network's initializer
  # MLP.new(key: value, key: value)
  def initialize(opts={})
    @learning_rate_param             = 0.2
    @propagation_function            = lambda { |x| 1/(1+Math.exp(-1*(x))) }
    @derivative_propagation_function = lambda { |y| y*(1-y) }
  end

  ##THE ALGORITHM (for a MLP with one hidden layer)
  # training_iteration  = 1
  # do # ... until the error term (or momentum) is satisfactorily minimized ceased
  # mlp.train_pattern(input: [1,1], output: [0])
  def train_pattern(training_set)
    # FEED FORWARD
    # 1. Apply input vectors to input neurons
    @training_inputs = vectorize(training_set[:input])
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
      result = (training[training_iteration][:output][j] - neuron.output)*(DERIVATIVE_OF_OUTPUT_ACTIVATION_FUNCTION_ON_NETINPUT)
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
