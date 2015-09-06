# COPYLEFT ORIGINAL AUTHOR ALEXANDER MARRS (github.com/marrsale / twitter.com/alx_mars)
class Neuron
  attr_accessor :weight, :net_input, :predecessors, :edges, :bias_weight

  # Neuron#new
  # Required arguments:
  # - input_node (Boolean): true if the neuron is in the input layer
  # - predecessors (Array): an array of neurons that precede the current one (will be empty for input layer neurons)
  # - successors (Array): an array of neurons that succeed the current one (will be empty for output layer neurons)
  # - activation_f (Proc): a lambda with the activation function (i.e. a logistic function for classification);
  #                        if none provided, there is a default
  #                        TODO: linear activation
  # - deriv_f (Proc): a lambda with the derivative of the activation function, used for gradient descent backpropagation;
  #                   if none provided, there is a default
  #                   TODO: incorporate the derivator method for generic derivations when none provided
  def initialize(opts={})
    self.bias_weight = opts[:bias] || 0.0

    set_edges! opts[:predecessors]

    # by default the activation function will be logistic
    self.activation_function = opts[:activation_f]  || -> x { 1/(1+Math.exp(-1*(x))) }
    self.activation_deriv    = opts[:deriv_f]       || -> y { y*(1-y) }
  end

  def input?
    predecessors.nil?
  end

  def output(x=net_input)
    (input?) ? x : activation_function.call(x)
  end

  def gradient
    activation_deriv.call(output)
  end

  def update_edge!(neuron, new_weight)
    if !(edges[neuron].nil?)
      self.edges[neuron] = new_weight
    else
      neuron.update_edge!(self, new_weight)
    end
  end

  def edge(neuron)
    self.edges[neuron] || neuron.edge(self)
  end

  def serialize
    if !(predecessors.nil?) && !(predecessors.empty?)
      predecessors.map do |pred|
        edge(pred)
      end
    else
      []
    end
  end

  def set_edges!(neighbors=nil)
    self.edges = {}
    self.predecessors = neighbors || predecessors
    unless input?
      predecessors.each { |pred| self.edges[pred] = initialize_edge }
    end
  end

  private

  attr_accessor :activation_function, :activation_deriv

  def initialize_edge(range=(-1.00..1.00))
    rand range
  end
end
