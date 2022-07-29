//#define LOG_MESSAGE

using System;
using Unity.MLAgents.SideChannels;

namespace RigidBody
{
    /// <summary>
    /// Custom side channel that can exchanges messages concerning the state.
    /// </summary>
    public class StateSideChannel : SideChannel
    {
        public BaxterState TrainState = BaxterState.None;

        public StateSideChannel()
        {
            ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            var receivedString = msg.ReadString();
#if LOG_MESSAGE
            Debug.Log("From Python : " + receivedString);
#endif
            TrainState = (BaxterState) Enum.Parse(typeof(BaxterState), receivedString);
        }

        /// <summary>
        /// Send a state
        /// </summary>
        /// <param name="state">The current state</param>
        public void SendState(string state)
        {
            using var msgOut = new OutgoingMessage();
            msgOut.WriteString(state);
            QueueMessageToSend(msgOut);
        }
    }
}