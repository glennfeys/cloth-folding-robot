using UnityEngine;

namespace RigidBody
{
    /// <summary>
    /// This class contains utility function for manipulating and reading joints.
    /// </summary>
    public static class JointUtils
    {
        private static readonly bool[] _mirror = {true, false, false, false};

        /// <summary>
        /// Get the current angle of the joint, normalized to the percentage of its articulation range.
        /// </summary>
        /// <param name="joint">The joint of which to get the normalized rotation</param>
        /// <returns>Float between 0 and 1, representing the joint's position in its articulation range</returns>
        public static float GetNormalizedJointRotation(ArticulationBody joint)
        {
            var xDrive = joint.xDrive;
            var lowerLimit = xDrive.lowerLimit;
            var upperLimit = xDrive.upperLimit;
            var currentRotation = GetJointRotation(joint);
            return (currentRotation - lowerLimit) / (upperLimit - lowerLimit);
        }

        /// <summary>
        /// Denormalize a rotation percentage according to the joint's articulation range,
        /// while clipping the angle at its boundary values.
        /// </summary>
        /// <param name="joint">The joint according to which the denormalization is calculated</param>
        /// <param name="percentage">The position in the articulation range of the joint</param>
        /// <returns>Denormalized angle</returns>
        public static float GetDenormalizedJointRotation(ArticulationBody joint, float percentage)
        {
            var xDrive = joint.xDrive;
            var lowerLimit = xDrive.lowerLimit;
            var upperLimit = xDrive.upperLimit;
            percentage = Mathf.Clamp01(percentage);
            return percentage * (upperLimit - lowerLimit) + lowerLimit;
        }

        /// <summary>
        /// Updates a joint's target rotation to a step in the specified direction.
        /// </summary>
        /// <param name="joint">Joint to update</param>
        /// <param name="partner">Mirrored partner of the joint</param>
        /// <param name="action">Direction the joint should move</param>
        /// <param name="stepSize">Step the joint should take in its movement</param>
        public static void UpdateJointRotation(ArticulationBody joint, ArticulationBody partner, int action,
            float stepSize)
        {
            var direction = action % 2 == 0 ? 1 : -1;

            var rotationChange = direction * stepSize;
            var target = GetDenormalizedJointRotation(joint, GetNormalizedJointRotation(joint) + rotationChange);
            RotateJointToTarget(joint, target);

            if (_mirror[action / 2])
                target = GetDenormalizedJointRotation(partner, GetNormalizedJointRotation(partner) - rotationChange);
            else
                target = GetDenormalizedJointRotation(partner, GetNormalizedJointRotation(partner) + rotationChange);

            RotateJointToTarget(partner, target);
        }

        /// <summary>
        /// Get the current rotation of a joint in degrees.
        /// </summary>
        /// <param name="joint">Joint of which to get the rotation</param>
        /// <returns>The current rotation of the joint</returns>
        public static float GetJointRotation(ArticulationBody joint)
        {
            var currentRotationRads = joint.jointPosition[0];
            var currentRotation = Mathf.Rad2Deg * currentRotationRads;
            return currentRotation;
        }

        /// <summary>
        /// Update a joint's target to a specified rotation.
        /// </summary>
        /// <param name="joint">Joint to rotate</param>
        /// <param name="target">Target rotation</param>
        public static void RotateJointToTarget(ArticulationBody joint, float target)
        {
            var drive = joint.xDrive;
            drive.target = target;
            joint.xDrive = drive;
        }

        /// <summary>
        /// Set the physical attributes of a joint.
        /// </summary>
        /// <param name="joint">Joint to set attributes on</param>
        /// <param name="stiffness">Joint stiffness</param>
        /// <param name="damping">Joint damping</param>
        public static void SetJointAttributes(ArticulationBody joint, float stiffness, float damping)
        {
            var drive = joint.xDrive;
            drive.stiffness = stiffness;
            drive.damping = damping;
            joint.xDrive = drive;
        }

        /// <summary>
        /// Reset a joint's simulation.
        /// </summary>
        /// <param name="joint">Joint to reset</param>
        public static void ResetJoint(ArticulationBody joint)
        {
            joint.jointPosition = new ArticulationReducedSpace(0f, 0f, 0f);
            joint.jointAcceleration = new ArticulationReducedSpace(0f, 0f, 0f);
            joint.jointForce = new ArticulationReducedSpace(0f, 0f, 0f);
            joint.jointVelocity = new ArticulationReducedSpace(0f, 0f, 0f);
            RotateJointToTarget(joint, 0.0f);
        }
    }
}