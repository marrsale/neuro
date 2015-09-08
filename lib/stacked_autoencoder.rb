# ORIGINAL AUTHOR ALEXANDER MARRS (github.com/marrsale / twitter.com/alx_mars)

# Stacked autoencoder
class ANN::StackedAutoEncoder < ANN::MLP
  def initialize(opts={})
    autoencoder_opts = opts.dup
    autoencoder_opts[:output] ||= opts[:input]

    super autoencoder_opts
  end

  def append_layer!(n=nil)
    n ||= hidden.last.size

    self.hidden_size << n
    self.num_layers += 1
    hidden << (create_layer n, hidden.last)
    output.each { |neuron| neuron.set_edges! hidden.last }
  end
end
