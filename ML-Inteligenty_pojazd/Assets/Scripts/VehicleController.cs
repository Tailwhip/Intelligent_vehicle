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
    private float visibleDistance = 50f;
    private float US1distance = 0f;
    private float US2distance = 0f;
    private float US3distance = 0f;
    private float US4distance = 0f;
    private float US5distance = 0f;

    /// Phototransistor sensors input 
    public PhototransistorMeassure phototransistor1;
    public PhototransistorMeassure phototransistor2;
    public PhototransistorMeassure phototransistor3;
    public PhototransistorMeassure phototransistor4;

    private bool collisionFail = false;
    private bool backFail = false;
    private bool win = false;
    private int saveTimer = 50000;
    private bool save = false;

    /// aNN definition:
    private int inputNumber = 4;
    private int outputNumber = 4;
    private int hiddenNumber = 1;
    private int hidNeuronsNumber = 7;
    private float n = 0.3f;

    /// aNN variables:
    private ANN ann;
    private List<float> calcOutputs;
    private List<float> states;
    private List<float> qs = new List<float>();
    public float sse = 0f;
    public float lastSSE = 1f;
    public float lastRewardSum = 1f;
    string currentWeights;

    List<float> Rewards = new List<float>();
    private float reward = 0.0f;                            //reward to associate with actions
    private float rewardSum = 0.0f;
    private float punishSum = 0.0f;
    private List<Replay> replayMemory = new List<Replay>(); //memory - list of past actions and rewards
    private int mCapacity = 10000;                          //memory capacity

    private float discount = 0.95f;                         //how much future states affect rewards
    private float exploreRate = 10f;                        //chance of picking random action
    private float maxExploreRate = 100.0f;					//max chance value
    private float minExploreRate = 0.05f;					//min chance value
    private float exploreDecay = 0.01f;
    private int resetTimer = 500;
    private int resetCounter = 0;
    private int reLearnCounter;
    private int rlcValue = 10000;
    private int rewardCounter = 0;

    /// objects:
    private Rigidbody rb;
    public GameObject wall;
    public GameObject LightSource;
    public Window_Graph graph;

    /// variables for reset the agent:
    private Quaternion vehicleStartRot;
    private Vector3 vehicleStartPos;
    private Vector3 lightStartPos;
    private Vector3 wallStartPos;
    private float resetFactor = 3.5f;                       // light bias value for reset training:

    /// variables to count delta intensity:
    private float deltaIntensity = 0f;
    private float intensityOld = 100f;
    private float intensity = 0f;
    private int deltaCounter = 1;

    /// For OnGUI to display
    private int failCount = 0;
    private float timer = 0;
    private float bestTime = 0;

    /// light intensity input:
    private Vector2 intensity1;
    private Vector2 intensity2;
    private Vector2 intensity3;
    private Vector2 intensity4;

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
        intensityOld = 0;
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


    private void Steer()
    {
        if (Input.GetKey("w"))
        {
            rb.AddForce(this.transform.forward * 200f);
            //this.transform.Translate(this.transform.forward * 2f);
        }

        if (Input.GetKey("s"))
        {
            //this.transform.position += this.transform.forward * -2f;
            rb.AddForce(this.transform.forward * -200f);
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
        /// counting light intensity inputs values:
        intensity1.Set((LightSource.transform.position.x - phototransistor1.transform.position.x), (LightSource.transform.position.z - phototransistor1.transform.position.z));
        intensity2.Set((LightSource.transform.position.x - phototransistor2.transform.position.x), (LightSource.transform.position.z - phototransistor2.transform.position.z));
        intensity3.Set((LightSource.transform.position.x - phototransistor3.transform.position.x), (LightSource.transform.position.z - phototransistor3.transform.position.z));
        intensity4.Set((LightSource.transform.position.x - phototransistor4.transform.position.x), (LightSource.transform.position.z - phototransistor4.transform.position.z));

        intensity = (1f - (intensity1.magnitude / 100f)) + (1f - (intensity2.magnitude / 100f)) + (1f - (intensity3.magnitude / 100f)) + (1f - (intensity4.magnitude / 100f));
        /*
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
        */

        /// creating states lists:
        states = new List<float>();

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
        states.Add(Mathf.Clamp(Round(1.0f - (intensity1.magnitude / 50f)), -1f, 1f));
        states.Add(Mathf.Clamp(Round(1.0f - (intensity2.magnitude / 50f)), -1f, 1f));
        states.Add(Mathf.Clamp(Round(1.0f - (intensity3.magnitude / 50f)), -1f, 1f));
        states.Add(Mathf.Clamp(Round(1.0f - (intensity4.magnitude / 50f)), -1f, 1f));
    }

    private void Drive()
    {
        /// counting timer:
        resetTimer--;
        timer += Time.deltaTime;

        qs.Clear();
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
            rb.AddForce(this.transform.forward * Mathf.Clamp(qs[maxQIndex], -1f, 1f) * 200f);
        }
        
        if (maxQIndex == 1)
        {
            rb.AddForce(this.transform.forward * Mathf.Clamp(qs[maxQIndex], 0f, 1f) * -200f);
        }
        
        //? turning:
        if (maxQIndex == 2)
        {
            this.transform.Rotate(0, Mathf.Clamp(qs[maxQIndex], -1f, 1f) * 2f, 0, 0);
        }
        
        if (maxQIndex == 3)
        {
            this.transform.Rotate(0, Mathf.Clamp(qs[maxQIndex], 0f, 1f) * -2f, 0, 0);
        }
        
        deltaCounter--;

        /// counting delta intensity and use it to punish or reward:
        if (deltaCounter == 0)
        {
            deltaCounter = 50;
            deltaIntensity = intensityOld - ((intensity1.magnitude + intensity2.magnitude + intensity3.magnitude + intensity4.magnitude) / 4);
            //Debug.Log("DETLA: " + deltaIntensity);
            if (deltaIntensity > 0.10f)
            {
                reward += 0.01f;
            }
            else
            {
                reward = -20f;
                backFail = true;
            }
            intensityOld = (intensity1.magnitude + intensity2.magnitude + intensity3.magnitude + intensity4.magnitude) / 4;
        }
        else
            reward += -0.001f;

        if (collisionFail)
        {
            reward += -1.0f;
        }

        if (intensity > resetFactor)
        {
            reward = 5.0f;
            win = true;
        }

        /// setting replay memory:
        Replay lastMemory = new Replay(states, reward);

        Rewards.Add(reward);
        
        if (replayMemory.Count > mCapacity)
            replayMemory.RemoveAt(0);

        replayMemory.Add(lastMemory);

        /// training through replay memory:
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

            if (timer > bestTime)
            {
                bestTime = timer;
            }

            if (collisionFail)
                failCount++;

            replayMemory.Clear();
            ResetVehicle();
            Rewards.Clear();
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
        deltaCounter = 20;
        intensityOld = 0.0f;
        win = false;
        timer = 0;
        collisionFail = false;
        backFail = false;
        resetCounter++;
        Debug.Log(Round(resetCounter / rlcValue) + "." + resetCounter + ".Total reward: " + Rewards.Sum());
        reward = 0;
        resetTimer = 500;
        //deltaIntensity = 100f;
        intensityOld = 100f;
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

    private void FixedUpdate()
    {
        UpdateInput();
        Drive();
        //Steer();
        saveTimer--;
        if (saveTimer == 0)
        {
            SaveWeightsToFile();
            Debug.Log("------------------------------------WEIGHTS_SAVED!-------------------------------------");
            saveTimer = 50000;
            exploreRate = 99f;
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
        return (float)System.Math.Round(x, 2, System.MidpointRounding.AwayFromZero);
    }

    private void OnGUI()
    {
        /*
        GUI.color = Color.red;
        GUI.Label(new Rect(25, 25, 250, 30), "US1: " + states[2]);
        GUI.Label(new Rect(25, 50, 250, 30), "US2: " + states[1]);
        GUI.Label(new Rect(25, 75, 250, 30), "US3: " + states[2]);
        GUI.Label(new Rect(25, 100, 250, 30), "US4: " + states[3]);
        GUI.Label(new Rect(25, 125, 250, 30), "US5: " + states[4]);
        */
        
        GUI.color = Color.blue;
        GUI.Label(new Rect(10, 25, 250, 30), "PT1: " + states[0]);
        GUI.Label(new Rect(10, 50, 250, 30), "PT2: " + states[1]);
        GUI.Label(new Rect(10, 75, 250, 30), "PT3: " + states[2]);
        GUI.Label(new Rect(10, 100, 250, 30), "PT4: " + states[3]);

        GUI.color = Color.black;
        for (int i = 0; i < ann.numHidden; i++)
        {
            int height = 25;
            for (int j = 0; j < ann.numNPerHidden; j++)
            {
                GUI.Label(new Rect(((i+1)*100), height, 250, 30), "N" + (j + 1) + ": " + ann.neuronValue[j]);
                height += 25;
            }
        }

        GUI.color = Color.red;
        /// NN output:
        GUI.Label(new Rect(300, 25, 250, 30), "Przód: " + qs[0]);
        GUI.Label(new Rect(300, 50, 250, 30), "Tył: " + qs[1]);
        GUI.Label(new Rect(300, 75, 250, 30), "Prawo: " + qs[2]);
        GUI.Label(new Rect(300, 100, 250, 30), "Lewo: " + qs[3]);
        /*
        GUI.color = Color.green;
        GUI.Label(new Rect(500, 25, 250, 30), "Fails: " + failCount);
        GUI.Label(new Rect(500, 50, 250, 30), "Decay Rate: " + exploreRate);
        GUI.Label(new Rect(500, 75, 250, 30), "Best Time: " + bestTime);
        GUI.Label(new Rect(500, 100, 250, 30), "Timer: " + timer);
        GUI.Label(new Rect(500, 125, 250, 30), "Time Scale: " + Time.timeScale);
        //GUI.Label(new Rect(500, 150, 250, 30), "Velx: " + states[0]);
        //GUI.Label(new Rect(500, 175, 250, 30), "Velz: " + states[1]);
        */
        
        GUI.Label(new Rect(500, 200, 250, 30), "eta: " + ann.eta);
        GUI.Label(new Rect(500, 225, 250, 30), "last SSE: " + lastSSE);
        GUI.Label(new Rect(500, 250, 250, 30), "delta light: " + deltaIntensity);
        if (save == true)
            GUI.Label(new Rect(500, 275, 250, 30), "Save mode [Z button]: ON");
        else
            GUI.Label(new Rect(500, 275, 250, 30), "Save mode [Z button]: OFF");
    }

}
