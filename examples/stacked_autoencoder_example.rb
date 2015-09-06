require '../neuro'

def train_and_print s, input_sets, n=10000
  # Train it to recognize all of our inputs
  n.times do |n|
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
end

# Generate some random inputs for it to learn
input_sets = []
10.times do
  input_sets << [[0,1].sample, [0,1].sample, [0,1].sample]
end

s = ANN::StackedAutoEncoder.new input: 3, hidden: 2

train_and_print s, input_sets

print "\n\n"
puts 'Appending a new layer!'
s.append_layer! 3

train_and_print s, input_sets
