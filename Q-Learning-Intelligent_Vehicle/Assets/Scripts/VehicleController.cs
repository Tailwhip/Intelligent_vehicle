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

    /// Ultrasonic sensors input
    public UltrasonicMeassure ultraSens1;
    public UltrasonicMeassure ultraSens2;
    public UltrasonicMeassure ultraSens3;
    public UltrasonicMeassure ultraSens4;
    public UltrasonicMeassure ultraSens5;

    /// Phototransistor sensors input 
    public PhototransistorMeassure phototransistor1;
    public PhototransistorMeassure phototransistor2;
    public PhototransistorMeassure phototransistor3;
    public PhototransistorMeassure phototransistor4;

    /// Velocity input
    public AccelerometerMeassure accelerometer;

    /// aNN definition
    private int inputNumber = 11;
    private int outputNumber = 4;
    private int hiddenNumber = 1;
    private int hidNeuronsNumber = 32;
    private float n = 0.3f;

    /// Variables for reward function
    private bool collisionFail = false;
    private bool backFail = false;
    private bool win = false;

    /// Variables to count delta intensity
    private float deltaIntensity;
    private float intensityOld = 0f;
    private float intensity = 0f;
    private int deltaCounter = 1;

    /// Q-Learning variables
    private ANN ann;
    private List<float> calcOutputs;
    private List<float> states;
    private List<float> qs = new List<float>();
    public float sse = 0f;
    public float lastSSE = 1f;
    public float lastRewardSum = 1f;

    List<float> Rewards = new List<float>();
    private float reward = 0.0f;                            //reward to associate with actions
    private List<Replay> replayMemory = new List<Replay>(); //memory - list of past actions and rewards
    private int mCapacity = 10000;                          //memory capacity

    private float discount = 0.95f;                         //how much future states affect rewards
    private float exploreRate = 10f;                        //chance of picking random action
    private float maxExploreRate = 100.0f;					//max chance value
    private float minExploreRate = 0.05f;					//min chance value
    private float exploreDecay = 0.01f;
    
    /// variables for saving ANN weights
    string currentWeights;
    private bool save = false;
    private int saveTimer = 50000;

    /// objects
    private Rigidbody rb;
    public GameObject wall;
    public GameObject LightSource;

    /// variables for reset the agent
    private Quaternion vehicleStartRot;
    private Vector3 vehicleStartPos;
    private Vector3 lightStartPos;
    private Vector3 wallStartPos;
    private float lightThreshold = 3.5f;                       //light bias value for reset training:
    private int resetTimer = 500;
    private int resetCounter = 0;

    /// relearn variables
    private int reLearnCounter;
    private int rlcValue = 10000;
    private int rewardCounter = 0;
    

    private void Start()
    {
        ann = new ANN(inputNumber, outputNumber, hiddenNumber, hidNeuronsNumber, n);
        /*
        if (File.Exists(Application.dataPath + "/weights.txt"))
        {
            LoadWeightsFromFile();
            exploreRate = 0.05f;
        }
        */

        /// initialise delta intensity variables:
        intensity = phototransistor1.intensity + phototransistor2.intensity + phototransistor3.intensity + phototransistor4.intensity;
        intensityOld = 0.0f;

        Time.timeScale = 1f;
        reLearnCounter = rlcValue;

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

        if (Input.GetKeyDown("z"))
            if (save == true)
                save = false;
            else
                save = true;
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

    private void ManualSteering()
    {
        if (Input.GetKey("w"))
        {
            rb.AddForce(this.transform.forward * 300f);
        }

        if (Input.GetKey("s"))
        {
            rb.AddForce(this.transform.forward * -300f);
        }

        if (Input.GetKey("a"))
        {
            this.transform.Rotate(0, -2f, 0, 0);
        }

        if (Input.GetKey("d"))
        {
            this.transform.Rotate(0, 2f, 0, 0);
        }

        RewardFunction();
    }

    private void UpdateInput()
    {
        /// creating states lists:
        states = new List<float>();

        /// sending inputs:
        states.Add(ultraSens1.distance);
        states.Add(ultraSens2.distance);
        states.Add(ultraSens3.distance);
        states.Add(ultraSens4.distance);
        states.Add(ultraSens5.distance);
        
        states.Add(accelerometer.velocityX);
        states.Add(accelerometer.velocityZ);
   
        states.Add(phototransistor1.intensity);
        states.Add(phototransistor2.intensity);
        states.Add(phototransistor3.intensity);
        states.Add(phototransistor4.intensity);
        
        /// updating total intensity value:
        intensity = phototransistor1.intensity + phototransistor2.intensity + phototransistor3.intensity + phototransistor4.intensity;
    }

    private void RewardFunction()
    {
        deltaCounter--;

        /// Counting delta intensity and use it to punish or reward
        if (deltaCounter == 0)
        {
            deltaCounter = 1;
            deltaIntensity = intensity - intensityOld;
            if (deltaIntensity >= 0.01f)
            {
                reward += 0.005f;
            }
            else
            {
                reward += -0.005f;
                //backFail = true;
            }
            intensityOld = intensity;
        }

        if (collisionFail)
        {
            reward = -1.0f;
        }

        if (intensity > lightThreshold)
        {
            reward = 1.0f;
            win = true;
        }
    }

    void ResetVehicle()
    {
        if (reLearnCounter == 0)
        {
            if (reward > 0f)
            {
                rewardCounter++;
            }
            else
            {
                reLearnCounter = rlcValue;

                ann = new ANN(inputNumber, outputNumber, hiddenNumber, hidNeuronsNumber, n);
                this.transform.position = vehicleStartPos;
                this.transform.rotation = vehicleStartRot;
                rb.velocity = new Vector3(0f, 0f, 0f);
                rb.angularVelocity = new Vector3(0f, 0f, 0f);
                LightSource.transform.position = lightStartPos;
                wall.transform.position = wallStartPos;
                wall.transform.Rotate(0f, 0f, 0f, 0f);
                wall.transform.localScale = new Vector3(15f, 30f, 4f);
                Debug.Log("---------------------------------------RESTART!------------------------------------------");

            }

            if (rewardCounter > 15)
            {
                SaveWeightsToFile();
                Debug.Log("------------------------------------WEIGHTS_SAVED!-------------------------------------");
                Debug.Break();
            }
        }
        else
            reLearnCounter--;

        this.transform.position = vehicleStartPos + new Vector3(Random.Range(-20, 20), 0, (Random.Range(-5, 5)));
        this.transform.rotation = vehicleStartRot;
        rb.velocity = new Vector3(0f, 0f, 0f);
        rb.angularVelocity = new Vector3(0f, 0f, 0f);
        LightSource.transform.position = lightStartPos + new Vector3(Random.Range(-20, 20), 0, (Random.Range(-20, 10)));
        wall.transform.position = wallStartPos + new Vector3(Random.Range(-10, 10), 0, 0);
        wall.transform.Rotate(0f, Random.Range(-180f, 180f), 0f, 0f);
        wall.transform.localScale = new Vector3(Random.Range(10f, 20f), 30f, 4f);
        //deltaCounter = 20f;
        intensityOld = 0.0f;
        win = false;
        collisionFail = false;
        backFail = false;
        resetCounter++;
        Debug.Log(Round(resetCounter / rlcValue) + "." + resetCounter + ".Total reward: " + Rewards.Sum());
        reward = 0;
        resetTimer = 500;
        //intensityOld = 100f;
    }

    private void QLearning()
    { 
        /// Counting timer
        resetTimer--;

        qs.Clear();
        qs = SoftMax(ann.CalcOutput(states));

        float maxQ = qs.Max();
        int maxQIndex = qs.ToList().IndexOf(maxQ);

        /// Counting exploring output values
        exploreRate = Mathf.Clamp(exploreRate - exploreDecay, minExploreRate, maxExploreRate);
        if(Random.Range(0, 100) < exploreRate)
            maxQIndex = Random.Range(0, 4);
        
        /// Moving the vehicle using output values
        /// Move forward
        if (maxQIndex == 0)
        {
            rb.AddForce(this.transform.forward * Mathf.Clamp(qs[maxQIndex], 0f, 1f) * 300f);
        }
        
        if (maxQIndex == 1)
        {
            rb.AddForce(this.transform.forward * Mathf.Clamp(qs[maxQIndex], 0f, 1f) * -300f);
        }
        
        /// Turning
        if (maxQIndex == 2)
        {
            this.transform.Rotate(0, Mathf.Clamp(qs[maxQIndex], 0f, 1f) * 2f, 0, 0);
        }
        
        if (maxQIndex == 3)
        {
            this.transform.Rotate(0, Mathf.Clamp(qs[maxQIndex], 0f, 1f) * -2f, 0, 0);
        }

        RewardFunction();

        /// Setting replay memory
        Replay lastMemory = new Replay(states, reward);

        Rewards.Add(reward);
        
        if (replayMemory.Count > mCapacity)
            replayMemory.RemoveAt(0);

        replayMemory.Add(lastMemory);

        /// Training through replay memory
        if (collisionFail || resetTimer == 0 || win || backFail)
        {
            List<float> QOld = new List<float>();
            List<float> QNew = new List<float>();
            
            for (int i = replayMemory.Count - 1; i >= 0; i--)
            {
                List<float> toutputsOld = new List<float>();                                        // List of actions at time [t] (present)
                List<float> toutputsNew = new List<float>();                                        // List of actions at time [t + 1] (future)
                toutputsOld = SoftMax(ann.CalcOutput(replayMemory[i].states));                      // Action in time [t] is equal to NN output for [i] step states in replay memory

                float maxQOld = toutputsOld.Max();                                                  // maximum Q value at [i] step is equal to maximum Q value in the list of actions in time [t]
                int action = toutputsOld.ToList().IndexOf(maxQOld);                                 // number of action (in list of actions at time [t]) with maximum Q value is setted
                QOld.Add(toutputsOld[action]);

                float feedback;
                if (i == replayMemory.Count - 1)                                                    // if it's the end of replay memory or if by any reason it's the end of the sequence (in this case
                {                                                                                   // it's collision fail, timer reset and getting into the source of light) then the  
                    feedback = replayMemory[i].reward;                                              // feedback (new reward) is equal to the reward in [i] replay memory, because it's the end of the
                }                                                                                   // sequence and there's no event after to count Bellman's equation

                else
                {
                    toutputsNew = SoftMax(ann.CalcOutput(replayMemory[i + 1].states));              // otherwise the action at time [t + 1] is equal to NN output for [i + 1] step states
                    maxQ = toutputsNew.Max();                                                       // in replay memory and then feedback is equal to the Bellman's Equation
                    feedback = (replayMemory[i].reward +
                        discount * maxQ);
                }
                QNew.Add(feedback);

                if (save == true)
                    SaveToFile(QOld, QNew, Rewards, "QValues");

                float thisError = 0f;
                currentWeights = ann.PrintWeights();

                toutputsOld[action] = feedback;                                                     // then the action at time [t] with max Q value (the best action) is setted as counted feedback
                List<float> calcOutputs = ann.Train(replayMemory[i].states, toutputsOld);           // value and it's used to train NN for [i] state
                for (int j = 0; j < calcOutputs.Count; j++)
                    thisError += (Mathf.Pow((toutputsOld[j] - calcOutputs[j]), 2));
                thisError = thisError / calcOutputs.Count;
                sse += thisError;
            }
            sse /= replayMemory.Count;

            if (lastRewardSum < Rewards.Sum())
            {
                //ann.LoadWeights(currentWeights);
                ann.eta = Mathf.Clamp(ann.eta - 0.001f, 0.1f, 0.4f);
            }
            else
            {
                ann.eta = Mathf.Clamp(ann.eta + 0.001f, 0.1f, 0.4f);
                lastRewardSum = Rewards.Sum();
            }

            replayMemory.Clear();
            ResetVehicle();
            Rewards.Clear();
        }
    }

    public void SaveToFile(List<float> qold, List<float> qnew, List<float> rewards, string filename)
    {
        string data = "";
        for (int i = 0; i < qold.Count; i++)
        {
            data += i + ";" + qold[i] + ";" + qnew[i] + ";" + rewards[i] + System.Environment.NewLine;
        }
        string path = Application.dataPath + "/" + filename + ".txt";
        StreamWriter wf = File.CreateText(path);
        wf.WriteLine(data);
        wf.Close();
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

    private void FixedUpdate()
    {
        UpdateInput();
        QLearning();
        //ManualSteering();
        saveTimer--;
        if (saveTimer == 0)
        {
            SaveWeightsToFile();
            Debug.Log("------------------------------------WEIGHTS_SAVED!-------------------------------------");
            saveTimer = 50000;
            exploreRate = 99f;
        }
    }
    
    float Round(float x)
    {
        return (float)System.Math.Round(x, 2, System.MidpointRounding.AwayFromZero);
    }

    private void OnGUI()
    {
        myGUI();
    }

    private void myGUI()
    {
        GUI.color = Color.red;
        GUI.Label(new Rect(25, 25, 250, 30), "US1: " + ultraSens1.distance);
        GUI.Label(new Rect(25, 50, 250, 30), "US2: " + ultraSens2.distance);
        GUI.Label(new Rect(25, 75, 250, 30), "US3: " + ultraSens3.distance);
        GUI.Label(new Rect(25, 100, 250, 30), "US4: " + ultraSens4.distance);
        GUI.Label(new Rect(25, 125, 250, 30), "US5: " + ultraSens5.distance);

        GUI.color = Color.yellow;
        GUI.Label(new Rect(300, 25, 250, 30), "Velocity X: " + accelerometer.velocityX);
        GUI.Label(new Rect(300, 50, 250, 30), "Velocity Z: " + accelerometer.velocityZ);

        GUI.color = Color.blue;
        GUI.Label(new Rect(150, 25, 250, 30), "PT1: " + phototransistor1.intensity);
        GUI.Label(new Rect(150, 50, 250, 30), "PT2: " + phototransistor2.intensity);
        GUI.Label(new Rect(150, 75, 250, 30), "PT3: " + phototransistor3.intensity);
        GUI.Label(new Rect(150, 100, 250, 30), "PT4: " + phototransistor4.intensity);

        GUI.color = Color.green;
        GUI.Label(new Rect(500, 25, 250, 30), "delta intensity: " + deltaIntensity);
        GUI.Label(new Rect(500, 50, 250, 30), "Intensity: " + intensity);

        GUI.color = Color.green;
        GUI.Label(new Rect(500, 200, 250, 30), "eta: " + ann.eta);
        GUI.Label(new Rect(500, 225, 250, 30), "last SSE: " + lastSSE);

        if (save == true)
            GUI.Label(new Rect(500, 275, 250, 30), "Save mode [Z button]: ON");
        else
            GUI.Label(new Rect(500, 275, 250, 30), "Save mode [Z button]: OFF");
        /*
        GUI.color = Color.black;
        for (int i = 0; i < ann.numHidden; i++)
        {
            int height = 25;
            for (int j = 0; j < ann.numNPerHidden; j++)
            {
                GUI.Label(new Rect(((i + 1) * 100), height, 250, 30), "N" + (j + 1) + ": " + ann.neuronValue[j]);
                height += 25;
            }
        }
        
        GUI.color = Color.red;
        /// NN output:
        GUI.Label(new Rect(300, 25, 250, 30), "Przód: " + qs[0]);
        GUI.Label(new Rect(300, 50, 250, 30), "Tył: " + qs[1]);
        GUI.Label(new Rect(300, 75, 250, 30), "Prawo: " + qs[2]);
        GUI.Label(new Rect(300, 100, 250, 30), "Lewo: " + qs[3]);
        */
    }
}
