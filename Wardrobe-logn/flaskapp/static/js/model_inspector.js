import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Global variables
let scene, camera, renderer, controls;
let modelContainer;

// Initialize the 3D scene
function init() {
    // Get container
    modelContainer = document.getElementById('modelContainer');
    
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(
        75, 
        modelContainer.clientWidth / modelContainer.clientHeight, 
        0.1, 
        1000
    );
    camera.position.z = 2;
    camera.position.y = 1;
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(modelContainer.clientWidth, modelContainer.clientHeight);
    renderer.shadowMap.enabled = true;
    modelContainer.appendChild(renderer.domElement);
    
    // Add orbit controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1;
    controls.maxDistance = 5;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 2, 1);
    directionalLight.castShadow = true;
    scene.add(directionalLight);
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Handle window resize
function onWindowResize() {
    if (!modelContainer) return;
    
    camera.aspect = modelContainer.clientWidth / modelContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(modelContainer.clientWidth, modelContainer.clientHeight);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Load and inspect the model
function inspectModel(modelPath) {
    const loader = new GLTFLoader();
    
    // Remove existing model if any
    scene.children.forEach(child => {
        if (child.type === 'Group') {
            scene.remove(child);
        }
    });
    
    // Create a list to display mesh names
    const meshList = document.getElementById('meshList');
    meshList.innerHTML = '';
    
    loader.load(
        modelPath,
        (gltf) => {
            const model = gltf.scene;
            
            // Scale and position the model
            model.scale.set(1, 1, 1);
            model.position.set(0, 0, 0);
            
            // Inspect all meshes
            model.traverse((child) => {
                if (child.isMesh) {
                    // Add mesh name to the list
                    const listItem = document.createElement('div');
                    listItem.textContent = child.name;
                    listItem.className = 'mesh-item';
                    listItem.onclick = () => highlightMesh(child);
                    meshList.appendChild(listItem);
                    
                    // Enable shadows
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            
            // Add model to scene
            scene.add(model);
            
            // Log mesh names to console
            console.log('Model loaded:', modelPath);
            console.log('Mesh names:');
            model.traverse((child) => {
                if (child.isMesh) {
                    console.log('-', child.name);
                }
            });
        },
        (xhr) => {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        (error) => {
            console.error('Error loading model:', error);
        }
    );
}

// Highlight a mesh when clicked
function highlightMesh(mesh) {
    // Reset all meshes to normal color
    scene.traverse((child) => {
        if (child.isMesh) {
            child.material.emissive.setHex(0x000000);
        }
    });
    
    // Highlight the selected mesh
    mesh.material.emissive.setHex(0x00ff00);
    
    // Log mesh details
    console.log('Selected mesh:', mesh.name);
    console.log('Material:', mesh.material);
    console.log('Geometry:', mesh.geometry);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Three.js scene
    init();
    
    // Add buttons to load models
    const buttonContainer = document.getElementById('buttonContainer');
    
    const maleButton = document.createElement('button');
    maleButton.textContent = 'Load Male Model';
    maleButton.onclick = () => inspectModel('/static/models/avatar/male.gltf');
    buttonContainer.appendChild(maleButton);
    
    const femaleButton = document.createElement('button');
    femaleButton.textContent = 'Load Female Model';
    femaleButton.onclick = () => inspectModel('/static/models/avatar/female.gltf');
    buttonContainer.appendChild(femaleButton);
}); 