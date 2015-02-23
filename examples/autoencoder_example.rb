require '../neuro'
# Generate some random inputs for it to learn
input_sets = []
10.times do
  input_sets << [[0,1].sample, [0,1].sample, [0,1].sample]
end

# Instantiate the autoencoder
autoencoder = ANN.new(input: 3, hidden: [2], output: 3)

# Train it to recognize all of our inputs
10000.times do |n|
  input_sets.each do |set|
    autoencoder.train_pattern!(input: set, output: set)
  end

  print "\rFor iteration \##{n}, error term is #{autoencoder.last_error_term}." if n != 0 && (n % 2000 == 0)
end

puts ''
# Print out the result of the autoencoder for each
input_sets.each do |set|
  puts "For input: #{set},\tOutput: #{autoencoder.evaluate!(set).map { |obj| obj.round }}"
end
