using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron {

	public int numInputs;
	public float bias;
	public float output;
	public float errorGradient;
	public List<float> weights = new List<float>();
	public List<float> inputs = new List<float>();

	public Neuron(int nInputs)
	{
        float weightRange = 2.4f / nInputs;
		bias = UnityEngine.Random.Range(-weightRange,weightRange);
		numInputs = nInputs;

		for(int i = 0; i < nInputs; i++)
			weights.Add(UnityEngine.Random.Range(-weightRange, weightRange));
	}
}
