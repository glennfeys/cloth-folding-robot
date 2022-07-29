using System;
using System.Xml.Serialization;

namespace Configuration
{
    [Serializable]
    [XmlRoot("Configuration")]
    public sealed class Configuration
    {
        [XmlElement("PhysicsWorld")] public PhysicsWorldConfiguration PhysicsWorld { get; set; }
    }
}