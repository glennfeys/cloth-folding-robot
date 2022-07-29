//#define LOG_MESSAGE

using Unity.MLAgents.SideChannels;
using System;

namespace RigidBody
{
    /// <summary>
    /// Custom side channel that can exchange messages concerning the mode of training.
    /// The standard mode is training mode.
    /// </summary>
    public class ModeSideChannel : SideChannel
    {
        public TrainingMode TrainingMode = TrainingMode.Multi;

        public ModeSideChannel()
        {
            ChannelId = new Guid("df1a3b81-8285-417e-a81f-ead61ff54977");
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            var receivedString = msg.ReadString();
#if LOG_MESSAGE
            Debug.Log("From Python : " + receivedString);
#endif
            TrainingMode = (TrainingMode) Enum.Parse(typeof(TrainingMode), receivedString);
        }
    }
}