using System.IO;
using System.Xml.Serialization;
using UnityEngine;

namespace Configuration
{
    public sealed class ConfigurationProvider : MonoBehaviour
    {
        [SerializeField] private string configurationFilePath = "defaultConfiguration.xml";
        public Configuration Configuration { get; private set; }

        private void Awake()
        {
            Configuration = ReadFromFile(configurationFilePath);
            ApplyGlobalConfiguration();
        }

        /// <summary>
        /// Applies global parameters from the configuration file.
        /// </summary>
        private void ApplyGlobalConfiguration()
        {
            Time.timeScale = Configuration.PhysicsWorld.TimeScale;
        }

        /// <summary>
        /// Reads the configuration file and creates a configuration object.
        /// </summary>
        /// <param name="configurationFilePath">The file path of the configuration file</param>
        /// <returns>The configuration object</returns>
        private static Configuration ReadFromFile(string configurationFilePath)
        {
            var serializer = new XmlSerializer(typeof(Configuration));
            var reader = new StreamReader(configurationFilePath);
            var configuration = (Configuration) serializer.Deserialize(reader);
            reader.Close();
            return configuration;
        }
    }
}