using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

public class PhototransistorMeassure : MonoBehaviour
{
    public RenderTexture lightChceckTexture;
    public float intensity;

    private float intensityValue;
    private RenderTexture tmpTexture;
    private RenderTexture previous;
    private Texture2D tmp2DTexture;
    private Color32[] colours;

    // variables to count delta intensity:
    public float deltaIntensity;
    public float intensityOld = 0;
    public int deltaCounter = 1;

    private void Start()
    {
        tmpTexture = RenderTexture.GetTemporary(lightChceckTexture.width, lightChceckTexture.height, 0,
            RenderTextureFormat.Default, RenderTextureReadWrite.Linear);
        tmp2DTexture = new Texture2D(lightChceckTexture.width, lightChceckTexture.height);
    }

    // Update is called once per frame
    void Update()
    {        
        Graphics.Blit(lightChceckTexture, tmpTexture);
        
        previous = RenderTexture.active;
        RenderTexture.active = tmpTexture;

        tmp2DTexture.ReadPixels(new Rect(0, 0, tmpTexture.width, tmpTexture.height), 0, 0);
        tmp2DTexture.Apply();
        
        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(tmpTexture);
        colours = tmp2DTexture.GetPixels32();

        intensityValue = 0;
        
        for(int i =0; i < colours.Length; i++)
        {
            intensityValue += (0.2126f * colours[i].r) + (0.7152f * colours[i].g) + (0.0722f * colours[i].b);
        }

        intensityValue -= 3118144.25f;
        intensityValue = intensityValue / 5100000;//4400000; // 3539682;
        //intensityValue = Round(intensityValue);

        intensity = Mathf.Clamp(intensityValue, 0, 1);
        
        deltaCounter--;
        if (deltaCounter == 0)
        {
            deltaCounter = 1;
            deltaIntensity = intensity - intensityOld;
            intensityOld = intensity;
        }
    }

    float Round(float x)
    {
        return (float)System.Math.Round(x, 2, System.MidpointRounding.AwayFromZero);
    }
}
