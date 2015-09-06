require_relative '../neuro'
require 'pry'

describe ANN::MLP do
  context 'instantiating an mlp' do
    let(:input_size) { rand 1..10 }
    let(:hidden_sizes) {
      [].tap do |arr|
        (rand 1..10).times do
          arr << rand(1..10)
        end
      end
    }
    let(:output_size) { rand 1..10 }
    let(:mlp) { ANN::MLP.new input: input_size, hidden: hidden_sizes, output: output_size }

    it 'the correct dimensions' do
      expect(mlp.input.size).to eq input_size

      hidden_layer_sizes = mlp.hidden.map { |layer| layer.size }
      expect(hidden_sizes).to eq hidden_sizes

      expect(mlp.output.size).to eq output_size
    end

    it 'the correct connectivity' do
      layers = (mlp.hidden + [mlp.output])
      layers.inject(mlp.input) do |prev_layer, layer|
        layer.each do |neuron|
          expect(neuron.predecessors).to eq prev_layer
        end
      end
    end
  end

  context 'utilizing an mlp' do
  end

  context 'serializing an mlp' do
  end
end