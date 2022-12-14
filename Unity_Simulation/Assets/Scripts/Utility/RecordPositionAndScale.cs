using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace Utility
{
    /// <summary>
    /// This component stores the current position on every physics update.
    /// This can be stored in a file upon pressing 'S' on the keyboard to replay later.
    /// </summary>
    public sealed class RecordPosition : MonoBehaviour
    {
        [SerializeField] private string fileName;
        private List<RecordedPositionAndScale> _positions;
        private bool _isRecording;

        private void Awake()
        {
            _positions = new List<RecordedPositionAndScale>();
            _isRecording = true;
        }

        private void FixedUpdate()
        {
            if (_isRecording)
            {
                var tr = transform;
                _positions.Add(new RecordedPositionAndScale(tr.localPosition, tr.localScale));

                if (Input.GetKeyDown(KeyCode.S))
                {
                    _isRecording = false;
                    var sb = new StringBuilder();
                    foreach (var rp in _positions)
                    {
                        void WriteVector(Vector3 vector)
                        {
                            sb.Append(vector.x);
                            sb.Append(' ');
                            sb.Append(vector.y);
                            sb.Append(' ');
                            sb.Append(vector.z);
                        }

                        WriteVector(rp.Position);
                        sb.Append(' ');
                        WriteVector(rp.Scale);
                        sb.AppendLine();
                    }

                    Debug.Log($"Saving result to {fileName}");
                    _positions.Clear();
                    File.WriteAllText(fileName, sb.ToString());
                }
            }
        }
    }
}