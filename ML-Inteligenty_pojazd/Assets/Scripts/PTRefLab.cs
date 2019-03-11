using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PTRefLab : MonoBehaviour {

    public RenderTexture lightChceckTexture;
    public float intensity;
    private RenderTexture tmpTexture;
    private RenderTexture previous;
    private Texture2D tmp2DTexture;
    private Color32[] colours;

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

        intensity = 0;

        for (int i = 0; i < colours.Length; i++)
        {
            intensity += (0.2126f * colours[i].r) + (0.7152f * colours[i].g) + (0.0722f * colours[i].b);
        }

        intensity -= 3118144.25f;
        intensity = intensity / 3539682;
        // For 0 light source distance: 9367802
        // For 450 light source distance: 3539682
        intensity = Round(intensity);
    }

    void OnGUI()
    {
        GUI.color = Color.yellow;
        GUI.Label(new Rect(1000, 25, 250, 30), "ReferenceIntensity: " + intensity);
    }

    float Round(float x)
    {
        return (float)System.Math.Round(x, 2, System.MidpointRounding.AwayFromZero);
    }
}
