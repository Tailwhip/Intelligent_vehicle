using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AccelerometerMeassure : MonoBehaviour
{
    public Rigidbody meassuredObj;
    private float velMax = 39f;
    private float accMax = 2400f;
    private float lastVel = 0.0f;

    public float velocityX = 0f;
    public float velocityZ = 0f;

    public float accelerationX = 0f;
    public float accelerationZ = 0f;

    // Update is called once per frame
    void FixedUpdate()
    {
        /// Counting velocity X and Z
        velocityX = Mathf.Clamp((meassuredObj.velocity.x / velMax), -1, 1);
        velocityZ = Mathf.Clamp((meassuredObj.velocity.z / velMax), -1, 1);
        /// Counting acceleration X and Z
        /*
        accelerationX = acceleration(velocityX);
        accelerationZ = acceleration(velocityZ);
        */    
    }

    /// Function for counting acceleration
    private float acceleration(float vel)
    {
        float deltaTime = vel - lastVel;
        lastVel = vel;

        return (vel / Time.fixedDeltaTime) / accMax;
    }
}
