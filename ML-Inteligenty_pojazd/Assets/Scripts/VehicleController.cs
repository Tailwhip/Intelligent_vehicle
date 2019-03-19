using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;

public class Replay
{
    public List<float> states;
    public float reward;

    public Replay(List<float> inputs, float r)
    {
        states = new List<float>();
        for (int i = 0; i < inputs.Count ; i++)
        {
            states.Add(inputs[i]);
        }
        reward = r;
    }
}

public class VehicleController : MonoBehaviour {

    // Wheels input
    public WheelCollider rightWheel_C, leftWheel_C, backWheel_C;
    public Transform rightWheel_T, leftWheel_T, backWheel_T;
    public GameObject LightSource;

    // Ultrasonic sensors input
    public UltrasonicMeassure ultraSens1;
    public UltrasonicMeassure ultraSens2;
    public UltrasonicMeassure ultraSens3;
    public UltrasonicMeassure ultraSens4;
    public UltrasonicMeassure ultraSens5;

    private float visibleDistance = 50f;
    private float US1distance = 0f;
    private float US2distance = 0f;
    private float US3distance = 0f;
    private float US4distance = 0f;
    private float US5distance = 0f;

    // Phototransistor sensors input 
    public PhototransistorMeassure phototransistor1;
    public PhototransistorMeassure phototransistor2;
    public PhototransistorMeassure phototransistor3;
    public PhototransistorMeassure phototransistor4;

    private bool collisionFail = false;
    private bool win = false;
    private int saveTimer = 25000;

    private ANN ann;
    private List<float> calcOutputs;
    private List<float> states;
    private List<float> qs;

    private float reward = 0.0f;                            //reward to associate with actions
    private float rewardSum = 0.0f;
    private float punishSum = 0.0f;
    private List<Replay> replayMemory = new List<Replay>(); //memory - list of past actions and rewards
    private int mCapacity = 10000;                          //memory capacity

    private float discount = 0.95f;                         //how much future states affect rewards
    private float exploreRate = 100.0f;                     //chance of picking random action
    private float maxExploreRate = 100.0f;					//max chance value
    private float minExploreRate = 0.05f;					//min chance value
    private float exploreDecay = 0.01f;
    private int resetTimer = 500;
    private int resetCounter = 0;

    // Objects:
    private Rigidbody rb;
    public GameObject wall;

    // variables for reset the agent:
    private Quaternion vehicleStartRot;
    private Vector3 vehicleStartPos;
    private Vector3 lightStartPos;
    private float looseCount = 0f;
    private float winCount = 0f;
    private float showReward = 0f;
    private Vector3 wallStartPos;
    private float resetFactor = 3.0f;

    // variables to count delta intensity:
    private float deltaIntensity = 0f;
    private float intensityOld = 0f;
    private float intensity = 0f;
    private int deltaCounter = 1;

    // For OnGUI to display
    private int failCount = 0;
    private float timer = 0;
    private float bestTime = 0;

    // light intensity input:
    private Vector2 intensity1;
    private Vector2 intensity2;
    private Vector2 intensity3;
    private Vector2 intensity4;

    private void Start()
    {
        ann = new ANN(4, 4, 2, 7, 0.2f);
        if (File.Exists(Application.dataPath + "/weights.txt"))
        {
            LoadWeightsFromFile();
            exploreRate = 0.05f;
        }
            
        vehicleStartPos = this.transform.position;
        vehicleStartRot = this.transform.rotation;
        intensityOld = 0;
        Time.timeScale = 1.0f;

        // initialise reset position variables:
        rb = this.GetComponent<Rigidbody>();
        vehicleStartRot = this.transform.rotation;
        vehicleStartPos = this.transform.position;
        lightStartPos = LightSource.transform.position;
        wallStartPos = wall.transform.position;
    }

    private void Update()
    {
        if (Input.GetKeyDown("space"))
            ResetVehicle();
        if (this.transform.rotation.z < -0.12f || this.transform.rotation.z > 0.12f)
        {
            ResetVehicle();
        }
        if (Input.GetKeyDown("0"))
            Time.timeScale = 1f;
        if (Input.GetKeyDown("1"))
            Time.timeScale = 10f;
        if (Input.GetKeyDown("2"))
            Time.timeScale = 20f;
        if (Input.GetKeyDown("3"))
            Time.timeScale = 30f;
        if (Input.GetKeyDown("4"))
            Time.timeScale = 40f;
        if (Input.GetKeyDown("5"))
            Time.timeScale = 100f;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == "Obstacle")
        {
            collisionFail = true;
        }

        if (collision.gameObject.tag == "Win")
        {
            win = true;
        }
    }


    private void Steer()
    {
        if (Input.GetKey("w"))
        {
            this.transform.position += this.transform.forward * 2f;
        }

        if (Input.GetKey("s"))
        {
            this.transform.position += this.transform.forward * -2f;
        }

        if (Input.GetKey("a"))
        {
            this.transform.Rotate(0, -2f, 0, 0);
        }

        if (Input.GetKey("d"))
        {
            this.transform.Rotate(0, 2f, 0, 0);
        }
    }

    private void UpdateInput()
    {
        // counting light intensity inputs values:
        intensity1.Set((LightSource.transform.position.x - phototransistor1.transform.position.x), (LightSource.transform.position.z - phototransistor1.transform.position.z));
        intensity2.Set((LightSource.transform.position.x - phototransistor2.transform.position.x), (LightSource.transform.position.z - phototransistor2.transform.position.z));
        intensity3.Set((LightSource.transform.position.x - phototransistor3.transform.position.x), (LightSource.transform.position.z - phototransistor3.transform.position.z));
        intensity4.Set((LightSource.transform.position.x - phototransistor4.transform.position.x), (LightSource.transform.position.z - phototransistor4.transform.position.z));

        intensity = (1f - (intensity1.magnitude / 100f)) + (1f - (intensity2.magnitude / 100f)) + (1f - (intensity3.magnitude / 100f)) + (1f - (intensity4.magnitude / 100f));

        US1distance = 0f;
        US2distance = 0f;
        US3distance = 0f;
        US4distance = 0f;
        US5distance = 0f;

    int layerMask = 1 << 11;
        RaycastHit hit;

        if (Physics.Raycast(this.transform.position, this.transform.forward, out hit, visibleDistance, layerMask))
        {
            Debug.DrawRay(this.transform.position, this.transform.forward * hit.distance, Color.red);
            //dist = 1 - hit.distance / visibleDistance;
            US1distance = hit.distance;
        }

        if (Physics.Raycast(this.transform.position, this.transform.right, out hit, visibleDistance, layerMask))
        {
            Debug.DrawRay(this.transform.position, this.transform.right * hit.distance, Color.red);
            //dist = 1 - hit.distance / visibleDistance;
            US2distance = hit.distance;
        }

        if (Physics.Raycast(this.transform.position, Quaternion.AngleAxis(45, Vector3.up) * -this.transform.right, out hit, visibleDistance, layerMask))
        {
            Debug.DrawRay(this.transform.position, Quaternion.AngleAxis(45, Vector3.up) * -this.transform.right * hit.distance, Color.red);
            //dist = 1 - hit.distance / visibleDistance;
            US3distance = hit.distance;
        }

        if (Physics.Raycast(this.transform.position, -this.transform.right, out hit, visibleDistance, layerMask))
        {
            Debug.DrawRay(this.transform.position, -this.transform.right * hit.distance, Color.red);
            //dist = 1 - hit.distance / visibleDistance;
            US4distance = hit.distance;
        }

        if (Physics.Raycast(this.transform.position, Quaternion.AngleAxis(-45, Vector3.up) * this.transform.right, out hit, visibleDistance, layerMask))
        {
            Debug.DrawRay(this.transform.position, Quaternion.AngleAxis(-45, Vector3.up) * this.transform.right * hit.distance, Color.red);
            //dist = 1 - hit.distance / visibleDistance;
            US5distance = 1 - hit.distance / visibleDistance;
        }
    }
      
    private void Drive()
    {
        /// counting timer:
        resetTimer--;
        timer += Time.deltaTime;

        /// creating states lists:
        states = new List<float>();
        qs = new List<float>();

        /// sending inputs:
        /*
        states.Add(Round(0.5f - US1distance / visibleDistance));
        states.Add(Round(0.5f - US2distance / visibleDistance));
        states.Add(Round(0.5f - US3distance / visibleDistance));
        states.Add(Round(0.5f - US4distance / visibleDistance));
        states.Add(Round(0.5f - US5distance / visibleDistance));

        states.Add(Round(rb.velocity.x / 100f));
        states.Add(Round(rb.velocity.z / 100f));
        */
        states.Add(Round(0.5f - (intensity1.magnitude / 100f)));
        states.Add(Round(0.5f - (intensity2.magnitude / 100f)));
        states.Add(Round(0.5f - (intensity3.magnitude / 100f)));
        states.Add(Round(0.5f - (intensity4.magnitude / 100f)));

        qs = SoftMax(ann.CalcOutput(states));

        float maxQ = qs.Max();
        int maxQIndex = qs.ToList().IndexOf(maxQ);

        /// counting exploring output values:
        exploreRate = Mathf.Clamp(exploreRate - exploreDecay, minExploreRate, maxExploreRate);
        if(Random.Range(0, 100) < exploreRate)
            maxQIndex = Random.Range(0, 4);

        /// moving the vehicle using output values:
        /// move forward:
        if (maxQIndex == 0)
        {
            //this.transform.position += this.transform.forward * Mathf.Clamp(qs[maxQIndex], -1f, 1f) * 2f;
            rb.AddForce(this.transform.forward * Mathf.Clamp(qs[maxQIndex], -1f, 1f) * 80f);
        }
        
        if (maxQIndex == 1)
        {
            //this.transform.position += this.transform.forward * Mathf.Clamp(qs[maxQIndex], -1f, 1f) * -2f;
            rb.AddForce(this.transform.forward * Mathf.Clamp(qs[maxQIndex], 0f, 1f) * -80f);
        }
        
        // turning:
        if (maxQIndex == 2)
        {
            this.transform.Rotate(0, Mathf.Clamp(qs[maxQIndex], -1f, 1f) * 2f, 0, 0);
        }
        
        if (maxQIndex == 3)
        {
            this.transform.Rotate(0, Mathf.Clamp(qs[maxQIndex], 0f, 1f) * -2f, 0, 0);
        }
        
        //Debug.Log("0: " + qs[0]);
        //Debug.Log("1: " + qs[1]);
        deltaCounter--;
        // counting delta intensity and use it to punish or reward:
        /*
        if (deltaCounter == 0)
        {
            deltaCounter = 1;
            deltaIntensity = intensity - intensityOld;
            if (deltaIntensity <= 0.0f)
            {
                reward += -0.005f;
                punishSum += reward;
                //Debug.Log("LOOSE1");
                looseCount++;
            }

            intensityOld = intensity;
        }
        */
        reward += -0.005f;
        if (collisionFail)
        {
            reward += -1.0f;
            punishSum += reward;
        }

        if (intensity > resetFactor)
        {
            reward = 1f;
            win = true;
        }

        // setting replay memory:
        Replay lastMemory = new Replay(states, reward);

        if (replayMemory.Count > mCapacity)
            replayMemory.RemoveAt(0);

        replayMemory.Add(lastMemory);

        // training through replay memory:
        if (collisionFail || resetTimer == 0 || win)
        {
            for (int i = replayMemory.Count - 1; i >= 0; i--)
            {
                List<float> toutputsOld = new List<float>();
                List<float> toutputsNew = new List<float>();
                toutputsOld = SoftMax(ann.CalcOutput(replayMemory[i].states));

                float maxQOld = toutputsOld.Max();
                int action = toutputsOld.ToList().IndexOf(maxQOld);

                float feedback;
                if (i == replayMemory.Count - 1 || collisionFail || resetTimer == 0 || win)
                {
                    feedback = replayMemory[i].reward;
                }
                    
                else
                {
                    toutputsNew = SoftMax(ann.CalcOutput(replayMemory[i + 1].states));
                    maxQ = toutputsNew.Max();
                    feedback = (replayMemory[i].reward +
                        discount * maxQ);
                }

                toutputsOld[action] = feedback;
                ann.Train(replayMemory[i].states, toutputsOld);
            }

            if (timer > bestTime)
            {
                bestTime = timer;
            }

            if (collisionFail)
                failCount++;

            replayMemory.Clear();
            ResetVehicle();
        }
    }

    void ResetVehicle()
    {
        this.transform.position = vehicleStartPos + new Vector3(Random.Range(-20, 20), 0, (Random.Range(-5, 5)));
        this.transform.rotation = vehicleStartRot;
        rb.velocity = new Vector3(0f, 0f, 0f);
        rb.angularVelocity = new Vector3(0f, 0f, 0f);
        LightSource.transform.position = lightStartPos + new Vector3(Random.Range(-20, 20), 0, (Random.Range(-20, 10)));
        wall.transform.position = wallStartPos + new Vector3(Random.Range(-10, 10), 0, 0);
        wall.transform.Rotate(0f, Random.Range(-180f,180f), 0f, 0f);
        wall.transform.localScale = new Vector3(Random.Range(10f, 20f), 30f, 4f);
        deltaCounter = 20;
        intensityOld = 0.0f;
        win = false;
        timer = 0;
        collisionFail = false;
        resetCounter++;
        Debug.Log(resetCounter + ". Reward = " + reward);
        reward = 0;
        resetTimer = 500;
        /*
        ultraSens1.deltaCounter = 20;
        ultraSens2.deltaCounter = 20;
        ultraSens3.deltaCounter = 20;
        ultraSens4.deltaCounter = 20;
        ultraSens5.deltaCounter = 20;
        ultraSens1.distanceOld = 0.0f;
        ultraSens2.distanceOld = 0.0f;
        ultraSens3.distanceOld = 0.0f;
        ultraSens4.distanceOld = 0.0f;
        ultraSens5.distanceOld = 0.0f;
        */
    }

    private void FixedUpdate()
    {
        UpdateInput();
        Drive();
        Steer();
        saveTimer--;
        if (saveTimer == 0)
        {
            SaveWeightsToFile();
            Debug.Log("------------------------------------WEIGHTS_SAVED!-------------------------------------");
            saveTimer = 25000;
        }
    }

    void SaveWeightsToFile()
    {
        string path = Application.dataPath + "/weights.txt";
        StreamWriter wf = File.CreateText(path);
        wf.WriteLine(ann.PrintWeights());
        wf.Close();
    }

    void LoadWeightsFromFile()
    {
        string path = Application.dataPath + "/weights.txt";
        StreamReader wf = File.OpenText(path);

        if (File.Exists(path))
        {
            string line = wf.ReadLine();
            ann.LoadWeights(line);
        }
    }

    List<float> SoftMax(List<float> values)
    {
        float max = values.Max();

        float scale = 0.0f;
        for (int i = 0; i < values.Count; ++i)
            scale += Mathf.Exp((float)(values[i] - max));

        List<float> result = new List<float>();
        for (int i = 0; i < values.Count; ++i)
            result.Add(Mathf.Exp((float)(values[i] - max)) / scale);

        return result;
    }

    float Map(float newfrom, float newto, float origfrom, float origto, float value)
    {
        if (value <= origfrom)
            return newfrom;
        else if (value >= origto)
            return newto;
        return (newto - newfrom) * ((value - origfrom) / (origto - origfrom)) + newfrom;
    }

    float Round(float x)
    {
        return (float)System.Math.Round(x, 1, System.MidpointRounding.AwayFromZero); ;
    }

    private void OnGUI()
    {
        /*
        GUI.color = Color.red;
        GUI.Label(new Rect(25, 25, 250, 30), "US1: " + states[0]);
        GUI.Label(new Rect(25, 50, 250, 30), "US2: " + states[1]);
        GUI.Label(new Rect(25, 75, 250, 30), "US3: " + states[2]);
        GUI.Label(new Rect(25, 100, 250, 30), "US4: " + states[3]);
        GUI.Label(new Rect(25, 125, 250, 30), "US5: " + states[4]);

        GUI.color = Color.blue;
        GUI.Label(new Rect(150, 25, 250, 30), "PT1: " + states[7]);
        GUI.Label(new Rect(150, 50, 250, 30), "PT2: " + states[8]);
        GUI.Label(new Rect(150, 75, 250, 30), "PT3: " + states[9]);
        GUI.Label(new Rect(150, 100, 250, 30), "PT4: " + states[10]);

        GUI.color = Color.green;
        GUI.Label(new Rect(300, 25, 250, 30), "Fails: " + failCount);
        GUI.Label(new Rect(300, 50, 250, 30), "Decay Rate: " + exploreRate);
        GUI.Label(new Rect(300, 75, 250, 30), "Best Time: " + bestTime);
        GUI.Label(new Rect(300, 100, 250, 30), "Timer: " + timer);
        GUI.Label(new Rect(300, 125, 250, 30), "Reward: " + reward);
        GUI.Label(new Rect(300, 150, 250, 30), "Time Scale: " + Time.timeScale);
        GUI.Label(new Rect(300, 200, 250, 30), "Delta light: " + deltaIntensity);
        GUI.Label(new Rect(300, 225, 250, 30), "Reward Sum: " + rewardSum);
        GUI.Label(new Rect(300, 250, 250, 30), "Punishment Sum: " + punishSum);
        GUI.Label(new Rect(300, 275, 250, 30), "Move forward: " + forward);
        GUI.Label(new Rect(300, 300, 250, 30), "Velx: " + states[5]);
        GUI.Label(new Rect(300, 325, 250, 30), "Velz: " + states[6]);
        */
    }

}
