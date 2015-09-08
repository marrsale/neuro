require_relative '../neuro'

describe ANN::StackedAutoEncoder do
  let(:autoencoder) { ANN::StackedAutoEncoder.new input: 2, hidden: 2 }

  context 'utilizing a stacked autoencoder' do
    it 'should append layers' do
      original_number_layers = autoencoder.num_layers
      expect {
        autoencoder.append_layer!
      }.to change{autoencoder.hidden.size}.by 1
      expect(autoencoder.num_layers).to_not eq original_number_layers
    end

    context 'after appending a layer' do
      before do
        autoencoder.append_layer!
      end

      it 'should have the correct connectivity' do
        layers = (autoencoder.hidden + [autoencoder.output])
        layers.inject(autoencoder.input) do |prev_layer, layer|
          layer.each do |neuron|
            expect(neuron.predecessors).to eq prev_layer
          end
        end
      end

      it 'should function normally' do
        expect {
          autoencoder.evaluate! [1,1]
          autoencoder.train_pattern! input: [1,1], output: [1,1]
          autoencoder.evaluate! [1,1]
        }.to_not raise_error
      end
    end
  end

  context 'serializing and deserializing a stacked autoencoder' do
    it 'should produce identical results' do
      original_result = autoencoder.evaluate! [1,1]
      new_autoencoder = ANN::StackedAutoEncoder.from_serialization autoencoder.serialize
      expect(new_autoencoder.class).to eq ANN::StackedAutoEncoder
      expect(new_autoencoder.evaluate!([1,1])).to eq original_result
    end
  end
end
