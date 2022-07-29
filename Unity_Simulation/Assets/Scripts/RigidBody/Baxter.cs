//#define PRINT_REWARD

using System.Collections.Generic;
using SoftBody;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents.SideChannels;
using System.Collections;

namespace RigidBody
{
    /// <summary>
    /// This class represents the ML-Agent that interacts with the environment.
    /// </summary>
    public sealed class Baxter : Agent
    {
        [Tooltip("For single stage training, do you want to artificially make the first fold?")] [SerializeField]
        private bool artificiallyMakeFold;

        [Tooltip("For single stage training, do you want the grabbers to hold the cloth?")] [SerializeField]
        private bool holdCloth;

        [SerializeField] private float stepSize = 0.0005f;
        [SerializeField] private float stiffness = 500f;
        [SerializeField] private float damping = 500f;

        [SerializeField] private GameObject baxter;
        [SerializeField] private RectangularCloth cloth;

        [SerializeField] private Transform sunGlasses;

        private List<ArticulationBody> _jointList;
        private List<ArticulationBody> _mirrorJointList;

        public BaxterHandGrab leftHandGrab;
        public BaxterHandGrab rightHandGrab;

        private BaxterState _currentState;
        private StateSideChannel _stateChannel;
        private ModeSideChannel _modeChannel;

        private bool _hasRotated;
        private bool _isWaiting;
        private bool _hasFinishedLoop;

        private void Start()
        {
            baxter = GameObject.FindWithTag("robot");
            cloth = GameObject.FindWithTag("cloth").GetComponent<RectangularCloth>();

            _stateChannel = new StateSideChannel();
            _modeChannel = new ModeSideChannel();

            SideChannelManager.RegisterSideChannel(_stateChannel);
            SideChannelManager.RegisterSideChannel(_modeChannel);

            _jointList = new List<ArticulationBody>();
            _mirrorJointList = new List<ArticulationBody>();

            var usefulJoints = new List<string>
                {"right_upper_shoulder", "right_lower_shoulder", "right_lower_elbow", "right_lower_forearm"};
            var mirrorJoints = new List<string>
                {"left_upper_shoulder", "left_lower_shoulder", "left_lower_elbow", "left_lower_forearm"};

            foreach (var joint in baxter.GetComponentsInChildren<ArticulationBody>())
            {
                JointUtils.SetJointAttributes(joint, stiffness, damping);
                if (joint.jointType == ArticulationJointType.RevoluteJoint)
                {
                    if (usefulJoints.Contains(joint.name)) _jointList.Add(joint);

                    if (mirrorJoints.Contains(joint.name)) _mirrorJointList.Add(joint);
                }
            }

            //Debug.Log($"joints amount: {_jointList.Count}");
        }

        /// <summary>
        /// Let Baxter express victory by animating his sunglasses.
        /// </summary>
        /// <returns>A coroutine iterator which causes the sunglasses animation.</returns>
        private IEnumerator ExpressVictory()
        {
            const float height = 2.5f;
            const int steps = 100;

            sunGlasses.position += sunGlasses.up * height;
            sunGlasses.gameObject.SetActive(true);

            for (int i = 0; i < steps; ++i)
            {
                sunGlasses.position -= sunGlasses.up / steps * height;
                yield return null;
            }

            yield return new WaitForSeconds(2f);

            sunGlasses.gameObject.SetActive(false);
        }

        public override void OnEpisodeBegin()
        {
            //Debug.Log("## OnEpisodeBegin ##");
            _currentState = _modeChannel.TrainingMode.Equals(TrainingMode.Multi)
                ? BaxterState.GrabCloth1
                : _stateChannel.TrainState;

            _hasFinishedLoop = false;

            _stateChannel.SendState(_currentState.ToString());

            // Reset the cloth
            cloth.ResetToInitialState();
            if (!holdCloth)
            {
                leftHandGrab.Detach();
                rightHandGrab.Detach();
                leftHandGrab.isOn = false;
                rightHandGrab.isOn = false;
            }

            _hasRotated = false;

            if (artificiallyMakeFold) cloth.MakeFold();

            // i.e reset the position of Baxter's joints & the cloth
            foreach (var joint in _jointList) JointUtils.ResetJoint(joint);

            foreach (var joint in _mirrorJointList) JointUtils.ResetJoint(joint);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Collect all observations concerning the joints
            foreach (var joint in _jointList) sensor.AddObservation(JointUtils.GetNormalizedJointRotation(joint));

            foreach (var joint in _mirrorJointList) sensor.AddObservation(JointUtils.GetNormalizedJointRotation(joint));

            // Collect all observations concerning the cloth
            // [0, 1, 2]: vector of the upper left point
            // [n, n + 1, n + 2]: vector n. Counting in a row major manner.
            foreach (var t in cloth.GetBones())
            {
                var relativePosition = baxter.transform.InverseTransformPoint(t.transform.position);
                sensor.AddObservation(relativePosition);
            }

            // this is the vertical distance of the lower elbow, this is added as an observation for           
            // grabCloth1, since there is a "penalty" in the reward if the lower elbow gets too close to the table.
            // if this penalty isn't in place, it will train itself to lower the elbow toward the table, blocking off any chance of 
            // grabbing the cloth at a wrong position (and thus losing) but still get an OK reward because it is close to the edges of the cloth,
            // but this is not what we want.
            var elbowY = _jointList[3].transform.position.y;
            sensor.AddObservation(elbowY);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            if (_isWaiting) return;

            var action = actionBuffers.DiscreteActions[0];

            if (_currentState != BaxterState.Fold1 && _currentState != BaxterState.Fold2 || action / 2 != 0)
                JointUtils.UpdateJointRotation(_jointList[action / 2], _mirrorJointList[action / 2], action, stepSize);

            var reward = GetReward();
#if PRINT_REWARD
            Debug.Log($"Reward: {reward}");
#endif
            SetReward(reward);

            // the reward is high enough so we can call the cloth folded now
            if (IsEpisodeSuccessful(reward))
            {
                Debug.Log($"Reward WIN: {reward}");
                SetReward(10000.0f);
                if (_currentState == _stateChannel.TrainState ||
                    _modeChannel.TrainingMode.Equals(TrainingMode.Single))
                    EndEpisode();
                else
                    NextState();
            }

            // cloth is scrambled and no longer fixable -> end episode
            else if (HasEpisodeFailed(reward))
            {
                Debug.Log($"Reward LOSS: {reward}");
                SetReward(-10000.0f);
                EndEpisode();
            }
        }

        /// <summary>
        /// Make a transition to the next state
        /// </summary>
        private void NextState()
        {
            var send = !_hasFinishedLoop;

            switch (_currentState)
            {
                case BaxterState.GrabCloth1:
                    leftHandGrab.isOn = true;
                    rightHandGrab.isOn = true;
                    _currentState = BaxterState.Fold1;
                    break;

                case BaxterState.Fold1:
                    _currentState = BaxterState.GrabCloth2;
                    StartCoroutine(ReleaseCloth());
                    break;

                case BaxterState.GrabCloth2:
                    // Grab cloth when we are close enough
                    leftHandGrab.isOn = true;
                    rightHandGrab.isOn = true;

                    _currentState = BaxterState.Fold2;
                    break;

                case BaxterState.Fold2:
                    if (!_hasFinishedLoop)
                    {
                        StartCoroutine(ReleaseCloth2());
                        if (!ReferenceEquals(sunGlasses, null) && !sunGlasses.gameObject.activeSelf)
                            StartCoroutine(ExpressVictory());
                        _hasFinishedLoop = true;
                    }
                    break;
            }

            if (send)
                _stateChannel.SendState(_currentState.ToString());
        }

        /// <summary>
        /// Make baxter release the cloth from its grabbers
        /// </summary>
        /// <returns>A coroutine iterator</returns>
        private IEnumerator ReleaseCloth()
        {
            _isWaiting = true;

            // Stabilization time
            yield return new WaitForSeconds(3f);

            leftHandGrab.isOn = false;
            rightHandGrab.isOn = false;
            // Detach cloth
            leftHandGrab.Detach();
            rightHandGrab.Detach();

            // Reset positions
            foreach (var joint in _jointList) JointUtils.ResetJoint(joint);

            foreach (var joint in _mirrorJointList) JointUtils.ResetJoint(joint);

            yield return new WaitForSeconds(0.1f);

            cloth.Rotate();
            _hasRotated = true;
            _isWaiting = false;
        }

        /// <summary>
        /// Make baxter release the cloth from its grabbers and move arms to the side
        /// </summary>
        /// <returns>A coroutine iterator</returns>
        private IEnumerator ReleaseCloth2()
        {
            _isWaiting = true;

            // Stabilization time
            yield return new WaitForSeconds(3f);

            // Move arms away from baxter
            const int action = 1;

            // Disable magnet on grabber
            leftHandGrab.isOn = false;
            rightHandGrab.isOn = false;

            // Detach cloth
            leftHandGrab.Detach();
            rightHandGrab.Detach();

            // Wait
            yield return new WaitForSeconds(0.1f);

            // Move to the side
            JointUtils.UpdateJointRotation(_jointList[action / 2], _mirrorJointList[action / 2], action, 0.1f);

            _isWaiting = false;
        }

        /// <summary>
        /// Determine whether an episode has failed
        /// </summary>
        /// <param name="reward">The reward of this episode</param>
        /// <returns>Boolean indicating whether the episode failed</returns>
        private bool HasEpisodeFailed(float reward)
        {
            return _currentState switch
            {
                BaxterState.Fold1 => cloth.CheckEndEpisodeFold1(0.6f),
                BaxterState.Fold2 => cloth.CheckEndEpisodeFold2(),
                BaxterState.GrabCloth1 => reward <= -1.2f &&
                                          (leftHandGrab.CanGrabSomething() || rightHandGrab.CanGrabSomething()),
                BaxterState.GrabCloth2 => reward <= -2.5f &&
                                          (leftHandGrab.CanGrabSomething() || rightHandGrab.CanGrabSomething()) &&
                                          _hasRotated,
                _ => true
            };
        }

        /// <summary>
        /// Get the reward based on the current state
        /// </summary>
        /// <returns>Reward of the episode of the current state</returns>
        private float GetReward()
        {
            return _currentState switch
            {
                BaxterState.Fold1 => cloth.GetRewardFold1(),
                BaxterState.Fold2 => cloth.GetRewardFold2(),
                BaxterState.GrabCloth1 => cloth.GetRewardGrabCloth1(leftHandGrab.transform.position,
                    rightHandGrab.transform.position),
                BaxterState.GrabCloth2 => cloth.GetRewardGrabCloth2(leftHandGrab, rightHandGrab),
                _ => 0.0f
            };
        }

        /// <summary>
        /// Determine whether an episode is successful enough.
        /// </summary>
        /// <param name="reward">The reward of this episode</param>
        /// <returns>Boolean indicating whether the episode is successful</returns>
        private bool IsEpisodeSuccessful(float reward)
        {
            return _currentState switch
            {
                BaxterState.Fold1 => reward > -15f,
                BaxterState.Fold2 => reward > -20.0f,
                BaxterState.GrabCloth1 => reward > -1.2f && leftHandGrab.CanGrabSomething() &&
                                          rightHandGrab.CanGrabSomething(),
                BaxterState.GrabCloth2 => reward > 70.0f && leftHandGrab.CanGrabSomething() &&
                                          rightHandGrab.CanGrabSomething() && _hasRotated,
                _ => false
            };
        }
    }
}