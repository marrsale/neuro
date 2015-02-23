require '../neuro'
# train an autoencoder
autoencoder = MLP.new(input: 3, hidden: [2], output: 3)
100000.times do |n|
  pattern = [[1,0].sample,[1,0].sample,[1,0].sample] # array of random bits
  autoencoder.train_pattern!(input: pattern, output: pattern)
  puts "\nFor iteration \##{n}, error term is #{autoencoder.last_error_term}.\n" if n != 0 && (n % 2000 == 0)
end

3.times do
  pattern = [[1,0].sample,[1,0].sample,[1,0].sample]
  puts "Pattern: #{pattern}\t#{autoencoder.evaluate!(pattern)}"
end
