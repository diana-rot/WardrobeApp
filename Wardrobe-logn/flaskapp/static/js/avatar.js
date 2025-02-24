import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

class AvatarSystem {
    constructor() {
        this.container = document.getElementById('avatar-container');
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.controls = null;
        this.avatar = null;
        this.clothes = null;

        this.init();
        this.setupLights();
        this.setupControls();
        this.setupEventListeners();
        this.animate();
    }

    init() {
        // Setup renderer
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setClearColor(0xf0f0f0);
        this.container.appendChild(this.renderer.domElement);

        // Setup camera
        this.camera.position.set(0, 1.6, 2);
        this.camera.lookAt(0, 1, 0);

        // Add ground plane
        const groundGeometry = new THREE.PlaneGeometry(10, 10);
        const groundMaterial = new THREE.MeshStandardMaterial({ color: 0xcccccc });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Load initial avatar
        this.loadAvatar('female');
    }

    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Directional light
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(-2, 4, 2);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        // Add hemispheric light
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.4);
        hemiLight.position.set(0, 20, 0);
        this.scene.add(hemiLight);
    }

    setupControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 5;
        this.controls.update();
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize(), false);

        // Setup UI controls
        document.getElementById('gender-select').addEventListener('change', (e) => {
            this.loadAvatar(e.target.value);
        });

        document.getElementById('skin-color').addEventListener('input', (e) => {
            this.updateSkinColor(e.target.value);
        });

        document.getElementById('hair-color').addEventListener('input', (e) => {
            this.updateHairColor(e.target.value);
        });

        document.getElementById('hair-style').addEventListener('change', (e) => {
            this.updateHairStyle(e.target.value);
        });
    }

    loadAvatar(gender) {
        const loader = new GLTFLoader();
        loader.load(`/api/avatar/${gender}`, (gltf) => {
            if (this.avatar) {
                this.scene.remove(this.avatar);
            }
            this.avatar = gltf.scene;
            this.avatar.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            this.scene.add(this.avatar);

            // Apply current customizations
            this.applyCurrentCustomizations();
        }, undefined, (error) => {
            console.error('Error loading avatar:', error);
        });
    }

    updateSkinColor(color) {
        if (this.avatar) {
            this.avatar.traverse((child) => {
                if (child.isMesh && child.name.includes('Skin')) {
                    child.material.color.setStyle(color);
                }
            });
        }
    }

    updateHairColor(color) {
        if (this.avatar) {
            this.avatar.traverse((child) => {
                if (child.isMesh && child.name.includes('Hair')) {
                    child.material.color.setStyle(color);
                }
            });
        }
    }

    updateHairStyle(style) {
        if (this.avatar) {
            // Hide all hair meshes
            this.avatar.traverse((child) => {
                if (child.isMesh && child.name.includes('Hair')) {
                    child.visible = false;
                }
            });

            // Show selected style
            const selectedHair = this.avatar.getObjectByName(`Hair_${style}`);
            if (selectedHair) {
                selectedHair.visible = true;
            }
        }
    }

    applyCurrentCustomizations() {
        const skinColor = document.getElementById('skin-color').value;
        const hairColor = document.getElementById('hair-color').value;
        const hairStyle = document.getElementById('hair-style').value;

        this.updateSkinColor(skinColor);
        this.updateHairColor(hairColor);
        this.updateHairStyle(hairStyle);
    }

    loadClothing(itemId) {
        const loader = new GLTFLoader();
        loader.load(`/api/clothes/${itemId}`, (gltf) => {
            if (this.clothes) {
                this.scene.remove(this.clothes);
            }
            this.clothes = gltf.scene;
            this.scene.add(this.clothes);
        });
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize the avatar system when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const avatarSystem = new AvatarSystem();

    // Load available clothes
    fetch('/api/clothes')
        .then(response => response.json())
        .then(clothes => {
            const clothesList = document.getElementById('clothes-list');
            clothes.forEach(item => {
                const div = document.createElement('div');
                div.className = 'clothes-item';
                div.innerHTML = `
                    <label>
                        <input type="checkbox" value="${item.id}">
                        ${item.name}
                    </label>
                `;
                div.querySelector('input').addEventListener('change', (e) => {
                    if (e.target.checked) {
                        avatarSystem.loadClothing(item.id);
                    }
                });
                clothesList.appendChild(div);
            });
        })
        .catch(error => console.error('Error loading clothes:', error));
});