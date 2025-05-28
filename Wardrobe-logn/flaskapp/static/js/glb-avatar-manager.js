/**
 * Complete GLB Avatar Manager with Fixed Path Generation
 * Save as: static/js/glb-avatar-manager.js
 *
 * Fixed to use correct filename format: {bodySize}_{height}.glb
 * Example: m_medium.glb, f_tall.glb, etc.
 */

class CustomizableGLBAvatarManager {
    constructor(options = {}) {
        this.container = options.container || document.getElementById('avatar-container');
        this.debug = options.debug || false;

        // Avatar state
        this.avatarModel = null;
        this.clothingItems = new Map();

        // Avatar configuration
        this.config = {
            gender: 'female',
            bodySize: 'm',
            height: 'medium',
            skinColor: 'light',
            hairType: 'short',
            hairColor: 'brown',
            eyeColor: 'brown'
        };

        // Color definitions for customization
        this.skinColors = {
            'light': 0xfdbcb4,
            'medium': 0xee9b82,
            'tan': 0xd08b5b,
            'dark': 0xae5d29,
            'darker': 0x8b4513,
            'darkest': 0x654321,
            'pink': 0xffb6c1,
            'olive': 0xddbea9
        };

        this.eyeColors = {
            'brown': 0x6B4423,
            'blue': 0x4A90E2,
            'green': 0x50C878,
            'light-blue': 0x87CEEB,
            'purple': 0x800080,
            'gray': 0x708090,
            'dark-green': 0x228B22,
            'hazel': 0xB8860B
        };

        this.hairColors = {
            'brown': 0x8B4513,
            'black': 0x000000,
            'blonde': 0xFFD700,
            'red': 0xDC143C,
            'auburn': 0xA0522D,
            'gray': 0xC0C0C0,
            'white': 0xFFFFFF,
            'pink': 0xFF69B4
        };

        // Store references to materials for easy modification
        this.avatarMaterials = {
            skin: [],
            eyes: [],
            hair: [],
            underwear: []
        };

        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Loaders
        this.gltfLoader = null;
        this.textureLoader = null;

        // Initialize
        this.init();
    }

    init() {
        console.log('🤖 Initializing GLB Avatar Manager...');
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLighting();
        this.setupControls();
        this.setupLoaders();
        this.setupGround();
        this.animate();

        // Load default avatar
        this.loadDefaultAvatar();

        console.log('✅ GLB Avatar Manager initialized');
    }

    setupScene() {
        this.scene = new THREE.Scene();
        // Professional background like Blender
        this.scene.background = new THREE.Color(0x393939);
    }

    setupCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(35, aspect, 0.1, 1000);
        this.camera.position.set(0, 1.6, 4);
        this.camera.lookAt(0, 1.2, 0);
    }

    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false
        });

        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // GLB-optimized settings
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        const existingCanvas = this.container.querySelector('canvas');
        if (existingCanvas) {
            existingCanvas.remove();
        }
        this.container.appendChild(this.renderer.domElement);
    }

    setupLighting() {
        // Professional studio lighting

        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);

        // Key light
        const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
        keyLight.position.set(-5, 10, 5);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 50;
        keyLight.shadow.camera.left = -10;
        keyLight.shadow.camera.right = 10;
        keyLight.shadow.camera.top = 10;
        keyLight.shadow.camera.bottom = -10;
        this.scene.add(keyLight);

        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(5, 5, 5);
        this.scene.add(fillLight);

        // Rim light
        const rimLight = new THREE.DirectionalLight(0xffffff, 0.6);
        rimLight.position.set(0, 5, -8);
        this.scene.add(rimLight);
    }

    setupControls() {
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.1;
            this.controls.maxPolarAngle = Math.PI * 0.8;
            this.controls.minDistance = 1;
            this.controls.maxDistance = 10;
            this.controls.target.set(0, 1, 0);
        }
    }

    setupLoaders() {
        this.textureLoader = new THREE.TextureLoader();

        if (typeof THREE.GLTFLoader !== 'undefined') {
            this.gltfLoader = new THREE.GLTFLoader();
            console.log('✅ GLTFLoader available');
        } else {
            console.error('❌ GLTFLoader not available');
            throw new Error('GLTFLoader is required for GLB files');
        }
    }

    setupGround() {
        // Professional grid like Blender
        const gridHelper = new THREE.GridHelper(20, 20, 0x555555, 0x444444);
        gridHelper.material.opacity = 0.6;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);

        // Shadow plane
        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.ShadowMaterial({
            opacity: 0.3
        });

        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    async loadDefaultAvatar() {
        console.log('🤖 Loading default GLB avatar...');
        try {
            await this.loadAvatarFromConfig(this.config);
            console.log('✅ Default avatar loaded successfully');
        } catch (error) {
            console.error('❌ Failed to load default avatar:', error);
            this.loadFallbackAvatar();
        }
    }

    async loadAvatarFromConfig(config) {
        const avatarPath = this.getAvatarPath(config);
        return this.loadGLBAvatar(avatarPath);
    }

    getAvatarPath(config) {
        const { gender, bodySize, height } = config;

        // FIXED: Use correct filename format: {bodySize}_{height}.glb
        // Examples: m_medium.glb, l_tall.glb, s_short.glb
        const fileName = `${bodySize}_${height}`;

        // Path structure: /static/models/makehuman/bodies/{gender}/{bodySize}_{height}.glb
        const fullPath = `/static/models/makehuman/bodies/${gender}/${fileName}.glb`;

        console.log(`🔗 Avatar path generated: ${fullPath}`);
        return fullPath;
    }

    async loadGLBAvatar(glbPath) {
        console.log(`🔄 Loading GLB avatar from: ${glbPath}`);

        try {
            // Remove existing avatar
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
                this.avatarModel = null;
            }

            // Clear material references
            this.avatarMaterials = {
                skin: [],
                eyes: [],
                hair: [],
                underwear: []
            };

            this.avatarModel = await this.loadGLB(glbPath);
            this.setupAvatarModel();
            this.scene.add(this.avatarModel);

            // Apply current configuration colors
            this.updateSkinColor(this.config.skinColor);
            this.updateEyeColor(this.config.eyeColor);
            this.updateHairColor(this.config.hairColor);

            console.log('✅ GLB avatar loaded successfully');
            return this.avatarModel;

        } catch (error) {
            console.error('❌ Error loading GLB avatar:', error);
            throw error;
        }
    }

    async loadGLB(glbPath) {
        return new Promise((resolve, reject) => {
            console.log('🔄 Loading GLB from:', glbPath);

            this.gltfLoader.load(
                glbPath,
                (gltf) => {
                    console.log('✅ GLB loaded successfully');

                    const model = gltf.scene;

                    // Process the model for customization
                    this.processGLBModel(model);

                    resolve(model);
                },
                (progress) => {
                    if (this.debug) {
                        const percentage = (progress.loaded / progress.total * 100).toFixed(1);
                        console.log(`📊 GLB loading progress: ${percentage}%`);
                    }
                },
                (error) => {
                    console.error('❌ GLB loading error:', error);
                    reject(error);
                }
            );
        });
    }

    processGLBModel(model) {
        console.log('🎨 Processing GLB model for customization...');

        model.traverse((child) => {
            if (child.isMesh) {
                console.log('🔍 Processing mesh:', child.name || 'unnamed');

                // Categorize materials for later customization
                this.categorizeMaterial(child);

                // Enable shadows
                child.castShadow = true;
                child.receiveShadow = true;

                // Ensure materials are properly configured
                if (child.material) {
                    if (child.material.map) {
                        child.material.map.colorSpace = THREE.SRGBColorSpace;
                    }
                    child.material.needsUpdate = true;
                }
            }
        });

        console.log('📊 Categorized materials:', {
            skin: this.avatarMaterials.skin.length,
            eyes: this.avatarMaterials.eyes.length,
            hair: this.avatarMaterials.hair.length,
            underwear: this.avatarMaterials.underwear.length
        });
    }

    categorizeMaterial(mesh) {
        const meshName = (mesh.name || '').toLowerCase();
        const material = mesh.material;

        if (!material) return;

        // Categorize based on mesh name - Updated for better detection
        if (meshName.includes('body') || meshName.includes('head') ||
            meshName.includes('arm') || meshName.includes('leg') ||
            meshName.includes('torso') || meshName.includes('neck') ||
            meshName.includes('base') || meshName.includes('skin') ||
            meshName.includes('face') || meshName.includes('hand') ||
            meshName.includes('foot')) {
            this.avatarMaterials.skin.push(material);
        }
        else if (meshName.includes('eye') && !meshName.includes('brow') &&
                 !meshName.includes('lash') && !meshName.includes('lid')) {
            this.avatarMaterials.eyes.push(material);
        }
        else if (meshName.includes('hair') || meshName.includes('scalp')) {
            this.avatarMaterials.hair.push(material);
        }
        else if (meshName.includes('bra') || meshName.includes('thong') ||
                 meshName.includes('underwear') || meshName.includes('french') ||
                 meshName.includes('panties') || meshName.includes('briefs')) {
            this.avatarMaterials.underwear.push(material);
        }
        else {
            // Default to skin if unsure - more conservative approach
            console.log(`⚠️ Unrecognized mesh '${meshName}', categorizing as skin`);
            this.avatarMaterials.skin.push(material);
        }
    }

    setupAvatarModel() {
        if (!this.avatarModel) return;

        // Calculate bounds and scale
        const box = new THREE.Box3().setFromObject(this.avatarModel);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        // GLB files usually come at the right scale, but adjust if needed
        let scale = 1.0;
        if (size.y > 3) {
            scale = 1.8 / size.y;
        } else if (size.y < 1) {
            scale = 1.8 / size.y;
        }

        this.avatarModel.scale.setScalar(scale);

        // Position at ground level
        this.avatarModel.position.y = -box.min.y * scale;
        this.avatarModel.position.x = -center.x * scale;
        this.avatarModel.position.z = -center.z * scale;

        console.log(`📏 Avatar scaled by ${scale.toFixed(2)}, positioned at (${this.avatarModel.position.x.toFixed(2)}, ${this.avatarModel.position.y.toFixed(2)}, ${this.avatarModel.position.z.toFixed(2)})`);
    }

    // Avatar customization methods
    async updateGender(gender) {
        console.log(`👤 Updating gender to: ${gender}`);
        this.config.gender = gender;
        await this.loadAvatarFromConfig(this.config);
    }

    async updateBodySize(bodySize) {
        console.log(`📏 Updating body size to: ${bodySize}`);
        this.config.bodySize = bodySize;
        await this.loadAvatarFromConfig(this.config);
    }

    async updateHeight(height) {
        console.log(`📐 Updating height to: ${height}`);
        this.config.height = height;
        await this.loadAvatarFromConfig(this.config);
    }

    updateSkinColor(skinColor) {
        this.config.skinColor = skinColor;

        const newColor = this.skinColors[skinColor] || this.skinColors['light'];

        console.log(`🎨 Changing skin color to: ${skinColor} (0x${newColor.toString(16)})`);

        // Update all skin materials
        this.avatarMaterials.skin.forEach((material, index) => {
            if (material.color) {
                material.color.setHex(newColor);
                material.needsUpdate = true;
                console.log(`  ✓ Updated skin material ${index}`);
            }
        });
    }

    updateEyeColor(eyeColor) {
        this.config.eyeColor = eyeColor;

        const newColor = this.eyeColors[eyeColor] || this.eyeColors['brown'];

        console.log(`👁️ Changing eye color to: ${eyeColor} (0x${newColor.toString(16)})`);

        // Update all eye materials
        this.avatarMaterials.eyes.forEach((material, index) => {
            if (material.color) {
                material.color.setHex(newColor);
                material.needsUpdate = true;
                console.log(`  ✓ Updated eye material ${index}`);
            }
        });
    }

    updateHairColor(hairColor) {
        this.config.hairColor = hairColor;

        const newColor = this.hairColors[hairColor] || this.hairColors['brown'];

        console.log(`💇 Changing hair color to: ${hairColor} (0x${newColor.toString(16)})`);

        // Update all hair materials
        this.avatarMaterials.hair.forEach((material, index) => {
            if (material.color) {
                material.color.setHex(newColor);
                material.needsUpdate = true;
                console.log(`  ✓ Updated hair material ${index}`);
            }
        });
    }

    updateHairType(hairType) {
        this.config.hairType = hairType;
        console.log(`✂️ Hair type set to: ${hairType}`);
        // For GLB files, different hair types would require different models
        // This could trigger a model reload with different hair geometry
        console.log('ℹ️ Note: Hair type changes in GLB require different model files');
    }

    // Configuration management
    getConfiguration() {
        return { ...this.config };
    }

    async setConfiguration(newConfig) {
        const oldConfig = { ...this.config };
        this.config = { ...this.config, ...newConfig };

        console.log('🔧 Setting new configuration:', newConfig);

        // Major changes require reloading GLB
        const majorChanges = ['gender', 'bodySize', 'height', 'hairType'];
        const needsReload = majorChanges.some(key =>
            oldConfig[key] !== newConfig[key] && newConfig[key] !== undefined
        );

        if (needsReload) {
            console.log('🔄 Major change detected, reloading avatar...');
            await this.loadAvatarFromConfig(this.config);
        } else {
            console.log('🎨 Minor change, updating materials...');
            // Just update colors
            if (oldConfig.skinColor !== newConfig.skinColor && newConfig.skinColor) {
                this.updateSkinColor(newConfig.skinColor);
            }
            if (oldConfig.eyeColor !== newConfig.eyeColor && newConfig.eyeColor) {
                this.updateEyeColor(newConfig.eyeColor);
            }
            if (oldConfig.hairColor !== newConfig.hairColor && newConfig.hairColor) {
                this.updateHairColor(newConfig.hairColor);
            }
        }
    }

    loadFallbackAvatar() {
        console.log('🔄 Loading fallback avatar...');

        const geometry = new THREE.CapsuleGeometry(0.3, 1.4, 4, 8);
        const material = new THREE.MeshLambertMaterial({
            color: this.skinColors[this.config.skinColor] || this.skinColors['light']
        });

        this.avatarModel = new THREE.Mesh(geometry, material);
        this.avatarModel.position.y = 0.9;
        this.avatarModel.castShadow = true;
        this.avatarModel.receiveShadow = true;

        this.scene.add(this.avatarModel);
        console.log('✅ Fallback avatar created');
    }

    // Clothing management
    clearAllClothing() {
        console.log('🗑️ Clearing all clothing items...');
        this.clothingItems.forEach((item, id) => {
            this.removeClothing(id);
        });
        this.clothingItems.clear();
        console.log('✅ All clothing items cleared');
    }

    removeClothing(itemId) {
        const item = this.clothingItems.get(itemId);
        if (item && item.model) {
            this.scene.remove(item.model);
            this.clothingItems.delete(itemId);
            console.log(`🗑️ Removed clothing item: ${itemId}`);
            return true;
        }
        console.warn(`⚠️ Clothing item not found: ${itemId}`);
        return false;
    }

    // Animation loop
    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update();
        }

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    // Window resize handler
    onWindowResize() {
        if (!this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);

        console.log(`📐 Resized to: ${width}x${height}`);
    }

    // Debug method to list all materials
    debugMaterials() {
        if (!this.avatarModel) {
            console.log('❌ No avatar loaded');
            return;
        }

        console.log('🔍 Avatar materials debug:');
        this.avatarModel.traverse((child) => {
            if (child.isMesh && child.material) {
                console.log(`- ${child.name || 'unnamed'}: ${child.material.type}`, child.material);
            }
        });
    }

    // Get expected file list for directory structure
    getExpectedFiles() {
        const genders = ['female', 'male'];
        const bodySizes = ['xs', 's', 'm', 'l', 'xl'];
        const heights = ['short', 'medium', 'tall'];

        const files = [];

        genders.forEach(gender => {
            bodySizes.forEach(bodySize => {
                heights.forEach(height => {
                    const fileName = `${bodySize}_${height}.glb`;
                    const fullPath = `/static/models/makehuman/bodies/${gender}/${fileName}`;
                    files.push({
                        gender,
                        bodySize,
                        height,
                        fileName,
                        fullPath
                    });
                });
            });
        });

        return files;
    }

    cleanup() {
        if (this.renderer) {
            this.renderer.dispose();
        }
        console.log('🧹 GLB Avatar Manager cleaned up');
    }
}

// Basic GLB Avatar Manager (simplified version)
class GLBAvatarManager extends CustomizableGLBAvatarManager {
    constructor(options = {}) {
        super(options);
    }
}

// Make globally available
window.CustomizableGLBAvatarManager = CustomizableGLBAvatarManager;
window.GLBAvatarManager = GLBAvatarManager;
window.MakeHumanAvatarManager = CustomizableGLBAvatarManager; // Default to GLB version

console.log('✅ GLB Avatar Manager classes loaded successfully');