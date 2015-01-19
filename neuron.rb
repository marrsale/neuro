# COPYLEFT ORIGINAL AUTHOR ALEXANDER MARRS (github.com/marrsale / twitter.com/alx_mars)
class Neuron
  attr_accessor :weight
  attr_reader :activation, :predecessors, :successors, :is_input, :bias_weight, :net_input

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
  def initialize(opts)
    # pass block for non-default activation function (takes two arguments, a weight and a network input)
    # initialize weight and references to predecessors/successors
    @bias_weight  = 0.00
    @input_node   = opts[:input_node] || false
    @predecessors = opts[:predecessors] # array of neurons
    @successors   = opts[:successors] # ''

    # initialize the edges
    # ... see Neuron#edge method for details
    @edges = {}
    if !(@input_node) # input nodes have no predecessors
      @predecessors.each do |pred|
        @edges[pred] = initialize_edge
      end

      # initialize the activation function and its derivative
      # by default the activation function will be logistic
      @activation_function = opts[:activation_f]  || -> x { 1/(1+Math.exp(-1*(x))) }
      @activation_deriv    = opts[:deriv_f]       || -> y { y*(1-y) }
      # TODO: if activation_f, but no deriv_f provided, call derivator(opts[:activation_f]) to create a lambda that is the approximate derivative
    end
  end


  def net_input=(x)
    @net_input = @activation = x
  end

  def output
    if input?
      # input nodes don't do any work
      return @net_input
    else
      # apply the logistic function onto the activation and return it
      @activation_function.call(@activation)
    end
  end

  # Method for returning the derivative of the activation function on the net input, to determine
  # the slope of function, or rate of change, which is a coefficient used to alter the learning rate
  def gradient
    @activation_deriv.call(output) # TODO: or do we call on netinput???
  end

  # Method for updating the weight of an edge between two neurons, when given the neuron this is connected to
  # and the new weight
  def update_edge!(neuron, new_weight)
    iff_neuron neuron
    if !(@edges[neuron].nil?)
      @edges[neuron] = new_weight
    else
      neuron.update_edge!(self, new_weight)
    end
  end

  # Return the edge between self and a connected neuron
  # note that a neuron only retains edges to its predecessor for simplicity
  # however, if one has two connected neurons the edge can be accessed from either direction:
  #   neuron1.edge(neuron2) #
  #   neuron2.edge(neuron1) # both calls return the same result
  def edge(neuron)
    iff_neuron neuron
    # the edge either belongs to self or the other neuron, go get it
    @edges[neuron] || neuron.edge(self)
  end

  def input?
    @input_node
  end

  private

  def iff_neuron(neuron)
    raise "A neuron was expected.  Received #{neuron.class.name} instead." unless neuron.is_a? Neuron
  end

  # returns a randomized edge weight for when the neuron is created
  # takes a range object, or defaults to a value between 0 and 1
  def initialize_edge(range=(-1.00..1.00))
    rand range
  end

  # TODO: debug? test? don't incorporate this until certain it's usable
  # returns a lambda that is the derivative function of a provided lambda argument
  # the derivative is required for gradient descent; if the
  # def derivator(func)
  #   if func.is_a? Proc
  #     (-> x {
  #       ((func.(x+1e-13) - (func.(x-1e-13))/(2e-13)
  #     })
  #   else
  #     raise "Neuron#derivator must be given a Proc as an argument; #{func.class.name} was provided"
  #   end
  # end
end
