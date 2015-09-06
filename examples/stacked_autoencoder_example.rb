require '../neuro'

class ANN::StackedAutoEncoder < ANN::MLP
  def initialize(opts={})
    autoencoder_opts = opts.dup
    autoencoder_opts[:output] = opts[:input] unless (opts[:output] || 1) <= opts[:input]

    super autoencoder_opts
  end

  def append_layer!(n=2)
    self.hidden_size << n
    self.num_layers += 1
    hidden << (create_layer n, hidden.last)
    output.each { |neuron| neuron.set_edges! hidden.last }
  end
end

# Generate some random inputs for it to learn
input_sets = []
10.times do
  input_sets << [[0,1].sample, [0,1].sample, [0,1].sample]
end

s = ANN::StackedAutoEncoder.new input: 3, hidden: 2

# Train it to recognize all of our inputs
# 10000.times do |n|
#   input_sets.each do |set|
#     s.train_pattern! input: set, output: set
#   end
#
#   print "\rFor iteration \##{n}, error term is #{s.last_error_term}." if n != 0 && (n % 2000 == 0)
# end
#
# puts ''
# # Print out the result of the autoencoder for each
# input_sets.each do |set|
#   puts "For input: #{set},\tOutput: #{s.evaluate!(set).map { |obj| obj.round }}"
# end

puts 'Appending a new layer!'
s.append_layer!

# Train it to recognize all of our inputs
10000.times do |n|
  input_sets.each do |set|
    s.train_pattern! input: set, output: set
  end

  print "\rFor iteration \##{n}, error term is #{s.last_error_term}." if n != 0 && (n % 2000 == 0)
end

puts ''
# Print out the result of the autoencoder for each
input_sets.each do |set|
  puts "For input: #{set},\tOutput: #{s.evaluate!(set).map { |obj| obj.round }}"
end
