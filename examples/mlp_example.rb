require '../neuro'
# Generate MLP with one hidden layer and train it to learn XOR
mlp = ANN.new(input: 2, hidden: [2], output: 1)
100000.times do |n|
  mlp.train_pattern!(input: [0,0], output: [0])
  mlp.train_pattern!(input: [0,1], output: [1])
  mlp.train_pattern!(input: [1,1], output: [0])
  mlp.train_pattern!(input: [1,0], output: [1])
  print "\rFor iteration \##{n}, error term is #{mlp.last_error_term}." if n != 0 && (n % 2000 == 0)
end

print "\n"
[[1,1],[1,0],[0,0],[0,1]].each do |input_set|
  puts "For input #{input_set}\t#{mlp.evaluate!(input_set)}"
end
