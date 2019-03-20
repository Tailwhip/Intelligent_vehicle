using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;

public class BallController : MonoBehaviour {

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
    public GameObject ball;
    private Rigidbody ballRb;
    private BallState ballBs;

    // variables for reset the agent:
    private Quaternion ballStartRot;
    private Vector3 ballStartPos;
    private Quaternion platformStartRot;
    private Vector3 platformStartPos;

    // For OnGUI to display
    private int failCount = 0;
    private float timer = 0;
    private float bestTime = 0;

    private void Start()
    {
        ann = new ANN(3, 2, 1, 7, 0.2f);
        if (File.Exists(Application.dataPath + "/BallWeights.txt"))
        {
            LoadWeightsFromFile();
            exploreRate = 0.05f;
        }

        ballBs = ball.GetComponent<BallState>();
        ballRb = ball.GetComponent<Rigidbody>();
        rb = this.GetComponent<Rigidbody>();

        // initialise reset position variables:
        ballStartPos = ball.transform.position;
        ballStartRot = ball.transform.rotation;
        platformStartPos = this.transform.position;
        platformStartRot = this.transform.rotation;
        Time.timeScale = 1.0f;
    }

    private void Update()
    {
        if (Input.GetKeyDown("space"))
            ResetVehicle();

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
        /// creating states lists:
        states = new List<float>();
        
        /// sending inputs:
        states.Add(this.transform.rotation.x);
        states.Add(ball.transform.position.z);
        states.Add(ballRb.angularVelocity.x);
    }
      
    private void Drive()
    {
        /// counting timer:
        resetTimer--;
        timer += Time.deltaTime;
        
        qs = new List<float>();
        qs = SoftMax(ann.CalcOutput(states));

        float maxQ = qs.Max();
        int maxQIndex = qs.ToList().IndexOf(maxQ);
        /*
        /// counting exploring output values:
        exploreRate = Mathf.Clamp(exploreRate - exploreDecay, minExploreRate, maxExploreRate);
        if(Random.Range(0, 100) < exploreRate)
            maxQIndex = Random.Range(0, 4);
        */
        /// moving the agent using output values:
        /// move forward:
        if (maxQIndex == 0)
        {
            this.transform.Rotate(Mathf.Clamp(qs[maxQIndex], 0f, 1f) * 2f, 0f, 0f);
        }

        if (maxQIndex == 1)
        {
            this.transform.Rotate(Mathf.Clamp(qs[maxQIndex], 0f, 1f) * -2f, 0f, 0f);
        }

        if (ballBs.dropped)
        {
            reward += -1f;
            win = true;
        }
 
        else
            reward += 0.01f;


        // setting replay memory:
        Replay lastMemory = new Replay(states, reward);

        if (replayMemory.Count > mCapacity)
            replayMemory.RemoveAt(0);

        replayMemory.Add(lastMemory);

        // training through replay memory:
        if (ballBs.dropped)
        {
            for (int i = replayMemory.Count - 1; i >= 0; i--)
            {
                List<float> toutputsOld = new List<float>();                                        // List of actions at time [t] (present)
                List<float> toutputsNew = new List<float>();                                        // List of actions at time [t + 1] (future)
                toutputsOld = SoftMax(ann.CalcOutput(replayMemory[i].states));                      // Action in time [t] is equal to NN output for [i] step states in replay memory

                float maxQOld = toutputsOld.Max();                                                  // maximum Q value at [i] step is equal to maximum Q value in the list of actions in time [t]
                int action = toutputsOld.ToList().IndexOf(maxQOld);                                 // number of action (in list of actions at time [t]) with maximum Q value is setted

                float feedback;
                if (i == replayMemory.Count - 1 || replayMemory[i].reward == -1)                    // if it's the end of replay memory or if by any reason it's the end of the sequence (in this case
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

                toutputsOld[action] = feedback;                                                     // then the action at time [t] with max Q value (the best action) is setted as counted feedback
                ann.Train(replayMemory[i].states, toutputsOld);                                     // value and it's used to train NN for [i] state
            }

            if (timer > bestTime)
            {
                bestTime = timer;
            }

            replayMemory.Clear();
            ResetVehicle();
        }
    }

    void ResetVehicle()
    {
        this.transform.position = platformStartPos;
        this.transform.rotation = platformStartRot;
        ball.transform.position = ballStartPos;
        ball.transform.rotation = ballStartRot;
        rb.velocity = new Vector3(0f, 0f, 0f);
        rb.angularVelocity = new Vector3(0f, 0f, 0f);
        ballRb.velocity = new Vector3(0f, 0f, 0f);
        ballRb.angularVelocity = new Vector3(0f, 0f, 0f);
        win = false;
        timer = 0;
        ballBs.dropped = false;
        collisionFail = false;
        resetCounter++;
        Debug.Log(resetCounter + ". Reward = " + reward);
        reward = 0;
        resetTimer = 500;
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
        string path = Application.dataPath + "/BallWeights.txt";
        StreamWriter wf = File.CreateText(path);
        wf.WriteLine(ann.PrintWeights());
        wf.Close();
    }

    void LoadWeightsFromFile()
    {
        string path = Application.dataPath + "/BallWeights.txt";
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

        GUI.color = Color.green;
        GUI.Label(new Rect(300, 25, 250, 30), "Fails: " + failCount);
        GUI.Label(new Rect(300, 50, 250, 30), "Decay Rate: " + exploreRate);
        GUI.Label(new Rect(300, 75, 250, 30), "Best Time: " + bestTime);
        GUI.Label(new Rect(300, 100, 250, 30), "Timer: " + timer);
        GUI.Label(new Rect(300, 125, 250, 30), "Reward: " + reward);
        GUI.Label(new Rect(300, 150, 250, 30), "Time Scale: " + Time.timeScale);
        GUI.Label(new Rect(300, 225, 250, 30), "Reward Sum: " + rewardSum);
        GUI.Label(new Rect(300, 250, 250, 30), "Punishment Sum: " + punishSum);
        
    }

}
