# ORIGINAL AUTHOR ALEXANDER MARRS (github.com/marrsale / twitter.com/alx_mars)

require 'json'

module ANN
end

# Multilayer perceptron (logistic by default)
class ANN::MLP
  attr_reader :last_error_term
  attr_accessor :input, :hidden, :output, :input_size, :hidden_size, :output_size, :num_layers, :learning_rate_param
  # The Artificial Neural Network Initializer
  # ANN::new()
  # ex. for an MLP with am input layer of size 2, hidden layer size 2, output layer size 1, by default logistic/classification propagation
  #   @mlp = ANN::MLP.new(input: 2, hidden: 2, output: 1)
  # ex. for an MLP with an input layer size=2, 2 hidden layers size=2, output layer size=1
  #   @mlp = ANN::MLP.new(input: 2, hidden: [2, 2], output: 1)
  def initialize(opts={})
    self.learning_rate_param = opts[:learning_rate] || 0.20

    self.input_size   = opts[:input]
    opts[:hidden] = [opts[:hidden]] unless opts[:hidden].is_a? Array
    self.hidden_size  = opts[:hidden] || [@input_size] # default: one hidden layer, same size as input layer
    self.output_size  = opts[:output] || @input_size
    self.num_layers   = opts[:hidden].count

    create_layers!
  end

  # Applies the input vector and feeds forward through the graph
  def evaluate!(input_set)
    # FEED FORWARD
    # 1. Apply input vectors to input neurons
    input.each.with_index do |neuron, j|
      neuron.net_input = input_set[j]
    end

    # 2, 3, 4.  Calculate the net input values to the hidden neurons
    #           Calculate the outputs from the hidden layer neurons
    #           Calculate the net input values for the output layer
    (hidden + [output]).each do |layer|
      layer.each do |neuron|
        sum = neuron.predecessors.inject(0) do |acc, pred|
          acc + pred.output * neuron.edge(pred)
        end
        neuron.net_input = neuron.bias_weight + sum
      end
    end

    # 5. Calculate the output values for the output layer
    output.map(&:output)
  end

  # Method used for training the network to a specific set of inputs
  def train_pattern!(training_set)
    # First have the network evaluate the training set input
    evaluate! training_set[:input]

    # BACKPROPAGATE ERRORS
    # 6. Calculate error terms for the output units
    errors = {}
    output.each.with_index do |neuron, j|
      errors[neuron] = (training_set[:output][j] - neuron.output)*neuron.gradient
    end

    # 7. Calculate the error terms for the hidden layer neurons
    hidden.reverse.inject(output) do |prev_layer, hidden_layer|
      hidden_layer.each do |neuron|
        err_sum = prev_layer.inject(0) do |sum, succ|
          sum += errors[succ]*(neuron.edge(succ))
        end
        errors[neuron] = neuron.gradient*err_sum
      end
    end

    # 8, 9. Update weights on output layer
    #       Update weights on hidden layers
    (hidden + [output]).reverse.each do |layer|
      layer.each do |neuron|
        neuron.predecessors.each do |pred|
          neuron.update_edge! pred, (neuron.edge(pred) + (learning_rate_param)*(errors[neuron])*(pred.output))
        end
      end
    end

    # 10. Calculate the error term; this is the metric for how well the network is learning
    err_sum = output.inject(0) do |sum, neuron|
      sum + (errors[neuron] ** 2)
    end
    @last_error_term = (err_sum/2)
  end

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
  def serialize
    marshall.to_json
  end

  # initializes a new neural network from a serialization object (or file)
  def self.from_serialization(json_ann)
    attrs = JSON.parse(json_ann) if json_ann.is_a? String

    # binding.pry

    # Create an ANN of correct dimensions
    ann = self.new input: attrs['input_size'], hidden: attrs['hidden_size'], output: attrs['output_size']

    # Because the newly initialized ANN has random weights, we want to replace these with the weights from our given attrs
    deserializable_layers = (attrs['hidden_layers'] + [attrs['output_layer']])
    (ann.hidden + [ann.output]).each.with_index do |layer, layer_index|
      layer.each.with_index do |neuron, neuron_index|
        neuron.predecessors.each.with_index do |pred, pred_index|
          neuron.update_edge! pred, deserializable_layers[layer_index][neuron_index][pred_index]
        end
      end
    end

    ann
  end

  def inspect
    "< input: #{input.size}, hidden: [#{hidden.map(&:size).join(', ')}], output: #{output.size} >"
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
        arr << create_layer((hidden_size[i]), (arr.last || input))
      end
    end
    self.output = create_layer output_size, hidden.last
  end
end
