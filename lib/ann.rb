# COPYLEFT ORIGINAL AUTHOR ALEXANDER MARRS (github.com/marrsale / twitter.com/alx_mars)
module ANN
end

# Artificial Neural Network
class ANN::MLP
  attr_reader :last_error_term
  attr_accessor :input, :hidden, :output, :input_size, :hidden_size, :output_size, :num_layers
  # The Artificial Neural Network Initializer
  # ANN::new()
  # ex. for an MLP with am input layer of size 2, hidden layer size 2, output layer size 1, by default logistic/classification propagation
  #   @mlp = ANN::MLP.new(input: 2, hidden: 2, output: 1)
  # ex. for an MLP with an input layer size=2, 2 hidden layers size=2, output layer size=1
  #   @mlp = ANN::MLP.new(input: 2, hidden: [2, 2], output: 1)
  def initialize(opts={})
    @learning_rate_param = opts[:learning_rate] || 0.20

    # initialize layers
    self.input_size   = opts[:input]
    opts[:hidden] = [opts[:hidden]] unless opts[:hidden].is_a? Array # wrap in array if only a single number is provided (i.e. one hidden layer)
    self.hidden_size  = opts[:hidden] || [@input_size] # default: one hidden layer, same size as input layer
    self.output_size  = opts[:output] || @input_size
    self.num_layers   = opts[:hidden].count

    create_layers!
  end

  # Applies the input vector and feeds forward through the graph
  # eg. for an instance of ANN, @mlp, that has been trained the XOR function:
  #   @mlp.evaluate([1,1]) # => [0.01958032622180445]
  #   @mlp.evaulate([0,1]) # => [0.9535374396032609]
  def evaluate!(input_set)
    # FEED FORWARD
    # 1. Apply input vectors to input neurons
    inputs = input_set
    input.each.with_index do |neuron, j|
      neuron.net_input = inputs[j]
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
    output.map(&:output)
  end

  # THE ALGORITHM
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
    # inject with successive arrays instead of just iterating on output errors
    hidden.reverse.each do |hidden_layer|
      hidden_errors[hidden_layer] = hidden_layer.map.with_index do |neuron, j|
        sum = output_errors.inject(0) do |acc, succ|
          binding.pry
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
  end

  # returns a hash object containing the full contents of the neural network
  def marshall
    {
      learning_rate_param: @learning_rate_param,
      input_size: @input_size,
      hidden_size: @hidden_size,
      output_size: @output_size,
      num_layers: @num_layers,
      input_layer: input.map(&:serialize),
      hidden_layers: (hidden.map { |hidden_layer| hidden_layer.map(&:serialize) }), # array of arrays of serialized neurons
      output_layer: output.map(&:serialize)
    }
  end

  # writes the marshalled network as json
  def serialize(type=:json)
    if type == :json
      marshall.to_json
    else
      raise "#{self} can only serialize as JSON data."
    end
  end

  # initializes a new neural network from a serialization object (or file)
  def self.from_serialization(json_ann)
    if json_ann.is_a? String
      attrs = JSON.parse(json_ann)
    end

    # Create an ANN of correct dimensions
    @ann = ANN::MLP.new(input: attrs['input_size'], hidden: attrs['hidden_size'], output: attrs['output_size'])

    # Because the newly initialized ANN has random weights, we want to replace these with the weights from our given attrs
    # Note: because each layer owns the edges between itself and its predecessor layer
    # Set the weights for the hidden layer(s)
    @ann.hidden.each_with_index do |hidden_layer, hidden_layer_index| # for each hidden layer
      hidden_layer.each_with_index do |neuron, hidden_neuron_index| # and each neuron in the hidden layer
        neuron.predecessors.each_with_index do |predecessor, predecessor_index| # for each predecessor
          neuron.update_edge!(predecessor, attrs['hidden_layers'][hidden_layer_index][hidden_neuron_index][predecessor_index])
        end
      end
    end

    # Set the weights for the output layer
    @ann.output.each_with_index do |neuron, neuron_index|
      neuron.predecessors.each_with_index do |predecessor, predecessor_index|
        neuron.update_edge!(predecessor, attrs['output_layer'][neuron_index][predecessor_index])
      end
    end
    return @ann
  end

  def inspect
    "input: #{input.size}, hidden: [#{hidden.map(&:size).join(', ')}], output: #{output.size}"
  end

  private

  def create_layer(size,predecessors=nil)
    [].tap do |arr|
      size.times do
          arr << Neuron.new(predecessors: predecessors)
      end
    end
  end

  def create_layers!
    self.input = create_layer input_size
    self.hidden = [].tap do |arr|
      num_layers.times do |i|
        arr << create_layer((hidden_size[i-1]), (arr.last || input))
      end
    end
    self.output = create_layer output_size, hidden.last
  end
end
