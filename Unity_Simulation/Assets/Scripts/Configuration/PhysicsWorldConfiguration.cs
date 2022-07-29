using System;
using System.ComponentModel;
using System.Xml.Serialization;
using SoftBody;
using UnityEngine;

namespace Configuration
{
    [Serializable]
    public sealed class PhysicsWorldConfiguration
    {
        [XmlElement("TimeScale")] public float TimeScale { get; set; }
        [XmlElement("ClothUsesGpuSimulation")] public bool ClothUsesGpuSimulation { get; set; }
        [XmlElement("DeltaTimeDivisor")] public uint DeltaTimeDivisor { get; set; }
        [XmlElement("GravityMultiplier")] public float GravityMultiplier { get; set; }
        [XmlElement("IntegrationType")] public IntegrationType IntegrationType { get; set; }
        [XmlElement("ElasticSpringConstant")] public float ElasticSpringConstant { get; set; }
        [XmlElement("ShearSpringConstant")] public float ShearSpringConstant { get; set; }
        [XmlElement("BendSpringConstant")] public float BendSpringConstant { get; set; }

        [XmlElement("MeshBasedElasticSpringConstant")]
        public float MeshBasedElasticSpringConstant { get; set; }

        [XmlElement("MeshBasedShearSpringConstant")]
        public float MeshBasedShearSpringConstant { get; set; }

        [XmlElement("SpringInverseMass")] public float SpringInverseMass { get; set; }

        [XmlElement("SpringDamping")] public float SpringDamping { get; set; }

        /// <summary>
        /// The ratio of the final to initial relative velocity between two objects after collision.
        /// </summary>
        [XmlElement("RestitutionConstant")]
        public float RestitutionConstant { get; set; }

        [XmlElement("FrictionConstant")] public float FrictionConstant { get; set; }

        public Vector3 Gravity => Physics.clothGravity * GravityMultiplier;

        /// <summary>
        /// Gets the spring constant for the given spring damper type.
        /// </summary>
        /// <param name="type">The spring damper type.</param>
        /// <returns>The spring constant.</returns>
        /// <exception cref="InvalidEnumArgumentException">An invalid spring damper type was given. This should never happen and means there is a missing case here.</exception>
        public float SpringConstantForType(SpringDamperType type)
        {
            return type switch
            {
                SpringDamperType.Elastic => ElasticSpringConstant,
                SpringDamperType.Shear => ShearSpringConstant,
                SpringDamperType.Bend => BendSpringConstant,
                SpringDamperType.MeshElastic => MeshBasedElasticSpringConstant,
                SpringDamperType.MeshShear => MeshBasedShearSpringConstant,
                _ => throw new InvalidEnumArgumentException()
            };
        }
    }
}