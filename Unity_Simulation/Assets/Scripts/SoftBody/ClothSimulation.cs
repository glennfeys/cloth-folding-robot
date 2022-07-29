//#define DEBUG_GIZMOS
//#define R_KEY_RESETS_CLOTH

using System.Collections.Generic;
using System.Linq;
using Configuration;
using SoftBody.Collision;
using SoftBody.Cpu;
using SoftBody.Gpu;
using Unity.Collections;
using UnityEngine;

namespace SoftBody
{
    /// <summary>
    /// Base class for every cloth simulation type,
    /// whether it is based on a mesh or a generated mesh at runtime.
    /// </summary>
    public abstract class ClothSimulation : MonoBehaviour
    {
        [SerializeField] private SphereCollider[] sphereColliders;
        [SerializeField] private BoxCollider[] cuboidColliders;
        [SerializeField] protected GameObject bonePrefab;
        [SerializeField] private ConfigurationProvider configurationProvider;
        [SerializeField] private float spatialHashingGridSize = 0.1f;
        [SerializeField] private ComputeShader computeShader;

        protected IClothSpringProcessor SpringProcessor;

        protected Transform[] Bones;

        private PhysicsWorldConfiguration PhysicsWorldConfiguration =>
            configurationProvider.Configuration.PhysicsWorld;

        protected virtual void Start()
        {
            if (PhysicsWorldConfiguration.ClothUsesGpuSimulation)
                SpringProcessor = new GpuClothSpringProcessor(PhysicsWorldConfiguration, Instantiate(computeShader));
            else
                SpringProcessor = new CpuClothSpringProcessor(PhysicsWorldConfiguration, spatialHashingGridSize);
        }

        /// <summary>
        /// Initializes the cloth simulation.
        /// </summary>
        /// <param name="mesh">The cloth mesh.</param>
        /// <param name="bones">The transforms of the bones.</param>
        protected void Initialize(Mesh mesh, Transform[] bones)
        {
            SpringProcessor.FinishInitialization();

            Bones = bones;

            mesh.bindposes = CalculateBindPosesFromTransforms(bones);

            var skinnedMeshRenderer = GetComponent<SkinnedMeshRenderer>();
            skinnedMeshRenderer.bones = bones;
            var bounds = mesh.bounds;
            var middleBone = bones[bones.Length / 2];
            bounds.center += transform.position - middleBone.position;
            skinnedMeshRenderer.localBounds = bounds;
            skinnedMeshRenderer.rootBone = middleBone;
            skinnedMeshRenderer.sharedMesh = mesh;
        }

        /// <summary>
        /// Enumerates spring nodes nearby a given sphere.
        /// </summary>
        /// <param name="centroid">The sphere centroid.</param>
        /// <param name="radius">The sphere radius.</param>
        /// <returns>An enumerable for the nearby spring nodes.</returns>
        public IEnumerable<ISpringNode> EnumerateNearbySphere(Vector3 centroid, float radius)
        {
            return SpringProcessor.EnumerateNearbySphere(centroid, radius);
        }

        /// <summary>
        /// Calculates the bind pose matrices from the bone transforms, which will be the inverse transformed positions.
        /// </summary>
        /// <param name="boneTransforms">An iterable over the bone transforms.</param>
        /// <returns>The bind pose matrices.</returns>
        private Matrix4x4[] CalculateBindPosesFromTransforms(IEnumerable<Transform> boneTransforms)
        {
            // Bone pose wants to have the inverse transformed position.
            var localToWorldMatrix = transform.localToWorldMatrix;
            return (from boneTransform in boneTransforms select boneTransform.worldToLocalMatrix * localToWorldMatrix)
                .ToArray();
        }

        private void FixedUpdate()
        {
            var sphereCollisionProxies = new NativeArray<ImmovableSphereCollisionAdapter>(sphereColliders.Length,
                Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var cuboidCollisionProxies = new NativeArray<ImmovableCuboidCollisionAdapter>(cuboidColliders.Length,
                Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            for (var i = 0; i < sphereColliders.Length; ++i)
                sphereCollisionProxies[i] = new ImmovableSphereCollisionAdapter(sphereColliders[i]);

            for (var i = 0; i < cuboidColliders.Length; ++i)
                cuboidCollisionProxies[i] = new ImmovableCuboidCollisionAdapter(cuboidColliders[i]);

            // Since we need an order inside FixedUpdate we can't just use the unordered execution order of 
            // FixedUpdate on different game objects.
            var deltaTime = Time.fixedDeltaTime;
            SpringProcessor.FixedUpdate(deltaTime, sphereCollisionProxies, cuboidCollisionProxies);

#if R_KEY_RESETS_CLOTH
            if (Input.GetKeyDown(KeyCode.R))
            {
                ResetToInitialState();
            }
#endif

            cuboidCollisionProxies.Dispose();
            sphereCollisionProxies.Dispose();
        }

#if DEBUG_GIZMOS
        private void OnDrawGizmos()
        {
            Gizmos.color = Color.red;
            SpringProcessor?.OnDrawGizmos();
        }
#endif

        private void OnDestroy()
        {
            SpringProcessor.OnDestroy();
        }

        /// <summary>
        /// Gets the bones of the cloth.
        /// </summary>
        /// <returns>The bones array.</returns>
        public Transform[] GetBones()
        {
            return Bones;
        }

        /// <summary>
        /// Resets the cloth to its initial start position
        /// </summary>
        public void ResetToInitialState()
        {
            SpringProcessor.ResetToInitialState();
        }
    }
}