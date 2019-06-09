using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

public class PhototransistorMeassure : MonoBehaviour
{
    public float intensity;
    private float intesityMax = 120f;
    private Vector2 intensVect = new Vector2(0f, 0f);
    public GameObject lightSource;

    void FixedUpdate()
    {
        /// light intensity counting:
        intensVect.Set((lightSource.transform.position.x - this.transform.position.x), (lightSource.transform.position.z - this.transform.position.z));

        intensity = Mathf.Clamp(1 - (intensVect.magnitude / intesityMax), 0, 1);
    }
}
