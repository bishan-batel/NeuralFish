using System.Collections.Generic;
using System.Drawing.Printing;
using System.Linq;
using Godot;

namespace NeuralFish.AI
{
	public class NeuralNetwork
	{
		public static float Sigmoid(float x) => 1 / (1 + Mathf.Exp(-x));

		public readonly int InputSize, OutputSize, HiddenLayerCount;
		public readonly int[] HiddenLayerSizes;

		// weights & biases
		public float[][] HiddenLayersWeights { get; }
		public float[][] HiddenLayersBiases { get; }
		public float[] OutputLayerWeights { get; }
		public float OutputLayerBias { get; }

		public NeuralNetwork(int inputSize, int outputSize, int[] hiddenLayerSizes, IReadOnlyList<float> serialized)
		{
			InputSize = inputSize;
			OutputSize = outputSize;
			HiddenLayerCount = hiddenLayerSizes.Length;
			HiddenLayerSizes = hiddenLayerSizes;

			HiddenLayersWeights = new float[HiddenLayerCount][];
			HiddenLayersBiases = new float[HiddenLayerCount][];

			var index = 0;

			// input layer
			index += InputSize;

			// hidden layers
			for (var i = 0; i < HiddenLayerCount; i++)
			{
				HiddenLayersWeights[i] = new float[HiddenLayerSizes[i]];
				HiddenLayersBiases[i] = new float[HiddenLayerSizes[i]];

				for (var j = 0; j < HiddenLayerSizes[i]; j++)
				{
					HiddenLayersWeights[i][j] = serialized[index++];
					HiddenLayersBiases[i][j] = serialized[index++];
				}
			}

			// output layer
			OutputLayerWeights = new float[OutputSize];
			for (var i = 0; i < OutputSize; i++)
			{
				OutputLayerWeights[i] = serialized[index++];
			}

			OutputLayerBias = serialized[index];
		}

		public NeuralNetwork(int inputSize, int outputSize, params int[] hiddenLayerSizes)
		{
			InputSize = inputSize;
			OutputSize = outputSize;
			HiddenLayerCount = hiddenLayerSizes.Length;
			HiddenLayerSizes = hiddenLayerSizes;

			HiddenLayersWeights = new float[HiddenLayerCount][];
			HiddenLayersBiases = new float[HiddenLayerCount][];

			var rng = new RandomNumberGenerator();

			// for each hidden layer initialize weights and biases
			for (var i = 0; i < HiddenLayerCount; i++)
			{
				HiddenLayersWeights[i] = new float[HiddenLayerSizes[i]];
				HiddenLayersBiases[i] = new float[HiddenLayerSizes[i]];

				// randomize weights and biases
				for (var j = 0; j < HiddenLayerSizes[i]; j++)
				{
					HiddenLayersWeights[i][j] = rng.Randf();
					HiddenLayersBiases[i][j] = rng.Randf();
				}
			}

			// output layer weights & biases
			OutputLayerWeights = new float[OutputSize];
			for (var i = 0; i < OutputSize; i++)
			{
				OutputLayerWeights[i] = rng.Randf();
			}

			OutputLayerBias = rng.Randf();
		}

		public float[] FeedForward(float[] input)
		{
			float[] prevLayer = input;

			// for each hidden layer
			for (var i = 0; i < HiddenLayerCount; i++)
			{
				var nextLayer = new float[HiddenLayerSizes[i]];

				// for each neuron in the layer
				for (var j = 0; j < HiddenLayerSizes[i]; j++)
				{
					// sum of the previous layer multiplied by the weights
					float sum = prevLayer.Sum(neuron => neuron * HiddenLayersWeights[i][j]);

					// add the bias
					sum += HiddenLayersBiases[i][j];

					// sigmoid activation function
					nextLayer[j] = Sigmoid(sum);
				}

				prevLayer = nextLayer;
			}

			// feed to output layer
			var output = new float[OutputSize];
			for (var i = 0; i < OutputSize; i++)
			{
				float sum = prevLayer.Sum(neuron => neuron * OutputLayerWeights[i]);
				sum += OutputLayerBias;
				output[i] = Sigmoid(sum);
			}

			return output;
		}

		public float[] Serialize()
		{
			var index = 0;
			var serialized = new float[InputSize + HiddenLayerCount * HiddenLayerSizes.Sum() * 2 + OutputSize + 1];

			// input layer
			index += InputSize;

			// hidden layers
			for (var i = 0; i < HiddenLayerCount; i++)
			{
				for (var j = 0; j < HiddenLayerSizes[i]; j++)
				{
					serialized[index++] = HiddenLayersWeights[i][j];
					serialized[index++] = HiddenLayersBiases[i][j];
				}
			}

			// output layer
			for (var i = 0; i < OutputSize; i++)
			{
				serialized[index++] = OutputLayerWeights[i];
			}

			serialized[index] = OutputLayerBias;

			return serialized;
		}
	}
}