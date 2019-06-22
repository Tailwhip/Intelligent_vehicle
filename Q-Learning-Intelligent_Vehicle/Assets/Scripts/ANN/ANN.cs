using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ANN {

    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden;
    public float M;
    public List<float> neuronValue = new List<float>();
    public float alpha = 0.9f;
    public float eta;
    private float wOld = 0f;
    List<Layer> layers = new List<Layer>();

    public ANN(int nI, int nO, int nH, int nPH, float n)
    {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        eta = n;

        if (numHidden > 0)
        {
            layers.Add(new Layer(numNPerHidden, numInputs));

            for (int i = 0; i < numHidden - 1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }

            layers.Add(new Layer(numOutputs, numNPerHidden));
        }
        else
        {
            layers.Add(new Layer(numOutputs, numInputs));
        }
    }

    public List<float> Train(List<float> inputValues, List<float> desiredOutput)
    {
        List<float> outputValues = new List<float>();
        outputValues = CalcOutput(inputValues, desiredOutput);
        UpdateWeights(outputValues, desiredOutput);
        return outputValues;
    }

    public List<float> CalcOutput(List<float> inputValues, List<float> desiredOutput)
    {
        List<float> inputs = new List<float>();
        List<float> outputValues = new List<float>();
        int currentInput = 0;

        if (inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numInputs);
            return outputValues;
        }

        inputs = new List<float>(inputValues);
        for (int i = 0; i < numHidden + 1; i++)
        {
            if (i > 0)
            {
                inputs = new List<float>(outputValues);
            }
            outputValues.Clear();

            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                float N = 0;
                layers[i].neurons[j].inputs.Clear();

                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[currentInput]);
                    N += layers[i].neurons[j].weights[k] * inputs[currentInput];
                    currentInput++;
                }

                N -= layers[i].neurons[j].bias;

                if (i == numHidden)
                    layers[i].neurons[j].output = ActivationFunctionO(N);
                else
                    layers[i].neurons[j].output = ActivationFunction(N);

                outputValues.Add(layers[i].neurons[j].output);
                currentInput = 0;
            }
        }
        return outputValues;
    }

    public List<float> CalcOutput(List<float> inputValues)
    {
        List<float> inputs = new List<float>();
        List<float> outputValues = new List<float>();
        int currentInput = 0;
        neuronValue.Clear();

        if (inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numInputs);
            return outputValues;
        }

        inputs = new List<float>(inputValues);
        for (int i = 0; i < numHidden + 1; i++)
        {
            if (i > 0)
            {
                inputs = new List<float>(outputValues);
            }
            outputValues.Clear();

            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                float N = 0;
                layers[i].neurons[j].inputs.Clear();

                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[currentInput]);
                    N += layers[i].neurons[j].weights[k] * inputs[currentInput];
                    currentInput++;
                }

                N -= layers[i].neurons[j].bias;

                if (i == numHidden)
                    layers[i].neurons[j].output = ActivationFunctionO(N);
                else
                {
                    layers[i].neurons[j].output = ActivationFunction(N);
                    neuronValue.Add(ActivationFunction(N));
                }
                    
                outputValues.Add(layers[i].neurons[j].output);
                currentInput = 0;
            }
        }
        return outputValues;
    }

    public string PrintWeights()
    {
        string weightStr = "";
        foreach (Layer l in layers)
        {
            foreach (Neuron n in l.neurons)
            {
                foreach (float w in n.weights)
                {
                    weightStr += w + ";";
                }
                weightStr += n.bias + ";";
            }
        }
        return weightStr;
    }

    public void LoadWeights(string weightStr)
    {
        if (weightStr == "") return;
        string[] weightValues = weightStr.Split(';');
        int w = 0;
        foreach (Layer l in layers)
        {
            foreach (Neuron n in l.neurons)
            {
                for (int i = 0; i < n.weights.Count; i++)
                {
                    n.weights[i] = System.Convert.ToSingle(weightValues[w]);
                    w++;
                }
                n.bias = System.Convert.ToSingle(weightValues[w]);
                w++;
            }
        }
    }

    void UpdateWeights(List<float> outputs, List<float> desiredOutput)
    {
        float error;
        for (int i = numHidden; i >= 0; i--)
        {
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                if (i == numHidden)
                {
                    error = desiredOutput[j] - outputs[j];
                    layers[i].neurons[j].errorGradient = AFDerivativeO(outputs[j]) * error;
                }
                else
                {
                    layers[i].neurons[j].errorGradient = AFDerivative(layers[i].neurons[j].output);
                    float errorGradSum = 0;
                    for (int p = 0; p < layers[i + 1].numNeurons; p++)
                    {
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    layers[i].neurons[j].errorGradient *= errorGradSum;
                }
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    //M = alpha * (layers[i].neurons[j].weights[k] - wOld);
                    if (i == numHidden)
                    {
                        error = desiredOutput[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += eta * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        //M = alpha * (layers[i].neurons[j].weights[k] - layers[i].neurons[j].weights[kOld]);
                        layers[i].neurons[j].weights[k] += eta * layers[i].neurons[j].inputs[k] *
                            layers[i].neurons[j].errorGradient;
                    }
                    wOld = layers[i].neurons[j].weights[k];
                }
                layers[i].neurons[j].bias += eta * -1 * layers[i].neurons[j].errorGradient;
            }
        }
    }

    //--------------------------------------------ACTIVATION-FUNCTIONS------------------------------------------------------  
	float ActivationFunction(float value)
	{
		return TanH(value);
	}

	float ActivationFunctionO(float value)
	{
        return Sigmoid(value);
	}

	float TanH(float value)
	{
        return (float)System.Math.Tanh(value);
    }

	float ReLu(float value)
	{
		if(value > 0) return value;
		else return 0;
	}

	float Linear(float value)
	{
		return value;
	}

	float LeakyReLu(float value)
	{
        if (value < 0) return Mathf.Clamp(0.01f * value, -1f, 0f);
   		else return Mathf.Clamp(value, 0f, 1f);
	}

	float Sigmoid(float value) 
	{
    	float k = (float) System.Math.Exp(value);
        k = k / (1.0f + k);
        if (value > 40)
            return 1;
        if (value < -40)
            return 0;
        return k;
    }

    //----------------------------------------ACTIVATION-FUNCTIONS-DERIVATIVES-----------------------------------------------------  
    float AFDerivative(float value)
    {
        return TanHDerivative(value);
    }

    float AFDerivativeO(float value)
    {
        return SigmoidDerivative(value);
    }

    float TanHDerivative(float value)
    {
        return 1 - value*value;
    }
    
    float ReLuDerivative(float value)
    {
        if (value < 0) return 0.0f;
        else return 1.0f;
    }
    
    float LeakyReLuDerivative(float value)
    {
        if (value < 0) return 0.01f;
        else return 1.0f;
    }

    float SigmoidDerivative(float value)
    {
        return value * (1 - value);
    }
}
