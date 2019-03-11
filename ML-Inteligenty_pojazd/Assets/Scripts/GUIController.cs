using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class GUIController : MonoBehaviour {

    public UltrasonicMeassure ultraSens1;
    public UltrasonicMeassure ultraSens2;
    public UltrasonicMeassure ultraSens3;
    public UltrasonicMeassure ultraSens4;
    public UltrasonicMeassure ultraSens5;

    public PhototransistorMeassure phototransistor1;
    public PhototransistorMeassure phototransistor2;
    public PhototransistorMeassure phototransistor3;
    public PhototransistorMeassure phototransistor4;

    private void OnGUI()
    {
        GUI.color = Color.red;
        GUI.Label(new Rect(25, 25, 250, 30), "US1: " + ultraSens1.distance);
        GUI.Label(new Rect(25, 50, 250, 30), "US2: " + ultraSens2.distance);
        GUI.Label(new Rect(25, 75, 250, 30), "US3: " + ultraSens3.distance);
        GUI.Label(new Rect(25, 100, 250, 30), "US4: " + ultraSens4.distance);
        GUI.Label(new Rect(25, 125, 250, 30), "US5: " + ultraSens5.distance);
        GUI.color = Color.blue;
        GUI.Label(new Rect(200, 25, 250, 30), "PT1: " + phototransistor1.intensity);
        GUI.Label(new Rect(200, 50, 250, 30), "PT2: " + phototransistor2.intensity);
        GUI.Label(new Rect(200, 75, 250, 30), "PT3: " + phototransistor3.intensity);
        GUI.Label(new Rect(200, 100, 250, 30), "PT4: " + phototransistor4.intensity);
    }
   
}
