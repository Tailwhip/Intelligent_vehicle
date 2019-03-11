using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UltrasonicMeassure : MonoBehaviour {

    private float visibleDistance = 105f;
    public float distance = 0f;
    public Transform ultrasonicSensor;
    private Vector3 toObstacle;

    // variables to count delta distance: 
    public float deltaDistance = 1f;
    public float distanceOld = 0;
    public int deltaCounter = 1;
    

    void Update()
    {
        deltaCounter--;
        if (deltaCounter == 0)
        {
            deltaCounter = 1;
            deltaDistance = distance - distanceOld;
            distanceOld = distance;
        }

        float dist = 0f;
        int layerMask = 1 << 11;
        RaycastHit hit;
        if (Physics.Raycast(ultrasonicSensor.position, ultrasonicSensor.transform.forward, out hit, visibleDistance, layerMask))
        {
            Debug.DrawRay(ultrasonicSensor.position, ultrasonicSensor.transform.forward * hit.distance, Color.red);
            dist = 1 - hit.distance / visibleDistance;
        }
        dist = Round(dist);
        distance = Mathf.Clamp(dist, 0, 1);
    }
   
    float Round(float x)
    {
        return (float)System.Math.Round(x, 2, System.MidpointRounding.AwayFromZero); ;
    }
  
}
