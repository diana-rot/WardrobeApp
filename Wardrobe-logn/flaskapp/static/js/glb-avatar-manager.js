/**
 * Enhanced GLB Avatar Manager with Eye Texture Loading
 * Save as: static/js/glb-avatar-manager.js
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

        // Eye texture mapping
        this.eyeTextures = {
            'brown': '/static/models/makehuman/bodies/female/textures/brown_eye.png',
            'blue': '/static/models/makehuman/bodies/female/textures/blue_eye.png',
            'green': '/static/models/makehuman/bodies/female/textures/green_eye.png',
            'light-blue': '/static/models/makehuman/bodies/female/textures/light_blue_eye.png',
            'purple': '/static/models/makehuman/bodies/female/textures/purple_eye.png',
            'gray': '/static/models/makehuman/bodies/female/textures/gray_eye.png',
            'grey': '/static/models/makehuman/bodies/female/textures/gray_eye.png',
            'dark-green': '/static/models/makehuman/bodies/female/textures/dark_green_eye.png',
            'hazel': '/static/models/makehuman/bodies/female/textures/hazel_eye.png'
        };

        // Loaded textures cache
        this.loadedTextures = new Map();

        // Color definitions
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
            'grey': 0x708090,
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

        // Material references
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
        this.gltfLoader = null;
        this.textureLoader = null;

        // Initialize
        this.init();
    }

    init() {
        console.log('🤖 Initializing GLB Avatar Manager with Eye Textures...');
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLighting();
        this.setupControls();
        this.setupLoaders();
        this.setupGround();
        this.animate();
        this.loadDefaultAvatar();
        console.log('✅ GLB Avatar Manager initialized');
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);
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
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
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
        keyLight.shadow.bias = -0.0001;
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
        fillLight.position.set(5, 5, 5);
        this.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xffffff, 0.8);
        rimLight.position.set(0, 5, -8);
        this.scene.add(rimLight);

        const topLight = new THREE.DirectionalLight(0xffffff, 0.4);
        topLight.position.set(0, 10, 0);
        this.scene.add(topLight);

        const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
        hemisphereLight.position.set(0, 20, 0);
        this.scene.add(hemisphereLight);

        const eyeLight = new THREE.DirectionalLight(0xffffff, 0.3);
        eyeLight.position.set(0, 2, 3);
        eyeLight.name = 'eyeLight';
        this.scene.add(eyeLight);
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
        const gridHelper = new THREE.GridHelper(20, 20, 0xcccccc, 0xdddddd);
        gridHelper.material.opacity = 0.8;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);

        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.ShadowMaterial({
            opacity: 0.15,
            color: 0x000000
        });

        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    async loadEyeTexture(eyeColor) {
        const texturePath = this.eyeTextures[eyeColor];

        if (!texturePath) {
            console.warn(`⚠️ No texture found for eye color: ${eyeColor}, falling back to brown`);
            return await this.loadEyeTexture('brown');
        }

        if (this.loadedTextures.has(eyeColor)) {
            console.log(`✅ Using cached eye texture: ${eyeColor}`);
            return this.loadedTextures.get(eyeColor);
        }

        return new Promise((resolve) => {
            console.log(`🔄 Loading eye texture: ${texturePath}`);

            this.textureLoader.load(
                texturePath,
                (texture) => {
                    console.log(`✅ Eye texture loaded: ${eyeColor}`);
                    texture.colorSpace = THREE.SRGBColorSpace;
                    texture.flipY = false;
                    texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
                    texture.minFilter = THREE.LinearFilter;
                    texture.magFilter = THREE.LinearFilter;
                    texture.generateMipmaps = false;
                    this.loadedTextures.set(eyeColor, texture);
                    resolve(texture);
                },
                (progress) => {
                    console.log(`📊 Loading texture progress: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
                },
                (error) => {
                    console.error(`❌ Failed to load eye texture: ${texturePath}`, error);
                    if (eyeColor !== 'brown') {
                        console.log(`🔄 Falling back to brown eye texture for ${eyeColor}`);
                        resolve(this.loadEyeTexture('brown'));
                    } else {
                        console.warn(`⚠️ Even brown eye texture failed, using color fallback`);
                        resolve(null);
                    }
                }
            );
        });
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
        const fileName = `${bodySize}_${height}`;
        const fullPath = `/static/models/makehuman/bodies/${gender}/${fileName}.glb`;
        console.log(`🔗 Avatar path generated: ${fullPath}`);
        return fullPath;
    }

    async loadGLBAvatar(glbPath) {
        console.log(`🔄 Loading GLB avatar from: ${glbPath}`);

        try {
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
                this.avatarModel = null;
            }

            this.avatarMaterials = {
                skin: [],
                eyes: [],
                hair: [],
                underwear: []
            };

            this.avatarModel = await this.loadGLB(glbPath);
            this.setupAvatarModel();
            this.scene.add(this.avatarModel);

            this.updateSkinColor(this.config.skinColor);
            await this.updateEyeColor(this.config.eyeColor);
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

        let eyeMeshCount = 0;
        let headMeshCount = 0;

        model.traverse((child) => {
            if (child.isMesh) {
                const meshName = (child.name || 'unnamed').toLowerCase();
                console.log('🔍 Processing mesh:', child.name || 'unnamed');

                if (meshName.includes('eye')) {
                    eyeMeshCount++;
                } else if (meshName.includes('head') || meshName.includes('face')) {
                    headMeshCount++;
                }

                this.categorizeMaterial(child);

                child.castShadow = true;
                child.receiveShadow = true;

                if (child.material) {
                    if (child.material.map) {
                        child.material.map.colorSpace = THREE.SRGBColorSpace;
                        child.material.map.flipY = false;
                    }
                    child.material.needsUpdate = true;
                }
            }
        });

        console.log('📊 Mesh Analysis:', {
            totalMeshes: model.children.length,
            eyeMeshes: eyeMeshCount,
            headMeshes: headMeshCount,
            materials: {
                skin: this.avatarMaterials.skin.length,
                eyes: this.avatarMaterials.eyes.length,
                hair: this.avatarMaterials.hair.length,
                underwear: this.avatarMaterials.underwear.length
            }
        });

        if (this.avatarMaterials.eyes.length === 0 && eyeMeshCount === 0) {
            console.warn('⚠️ No eye materials or meshes detected, creating fallback eyes...');
            setTimeout(() => this.createTexturedFallbackEyes(), 1000);
        }
    }

    categorizeMaterial(mesh) {
        const meshName = (mesh.name || '').toLowerCase();
        const materialName = (mesh.material.name || '').toLowerCase();
        const material = mesh.material;

        if (!material) return;

        console.log(`🏷️ Categorizing: Mesh="${meshName}" Material="${materialName}"`);

        if (meshName.includes('eye') || materialName.includes('eye')) {
            console.log(`👁️ Found eye material: ${meshName} | ${materialName}`);
            this.avatarMaterials.eyes.push(material);
        }
        else if (meshName.includes('head') || meshName.includes('face') ||
                 materialName.includes('head') || materialName.includes('face')) {
            console.log(`👤 Found head/face material: ${meshName} | ${materialName}`);
            this.avatarMaterials.eyes.push(material);
        }
        else if (meshName.includes('body') || meshName.includes('arm') ||
                 meshName.includes('leg') || meshName.includes('torso') ||
                 meshName.includes('neck') || meshName.includes('skin') ||
                 meshName.includes('hand') || meshName.includes('foot')) {
            this.avatarMaterials.skin.push(material);
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
            console.log(`❓ Unrecognized mesh '${meshName}', categorizing as skin`);
            this.avatarMaterials.skin.push(material);
        }
    }

    setupAvatarModel() {
        if (!this.avatarModel) return;

        const box = new THREE.Box3().setFromObject(this.avatarModel);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        let scale = 1.0;
        if (size.y > 3) {
            scale = 1.8 / size.y;
        } else if (size.y < 1) {
            scale = 1.8 / size.y;
        }

        this.avatarModel.scale.setScalar(scale);
        this.avatarModel.position.y = -box.min.y * scale;
        this.avatarModel.position.x = -center.x * scale;
        this.avatarModel.position.z = -center.z * scale;

        console.log(`📏 Avatar scaled by ${scale.toFixed(2)}, positioned at (${this.avatarModel.position.x.toFixed(2)}, ${this.avatarModel.position.y.toFixed(2)}, ${this.avatarModel.position.z.toFixed(2)})`);
    }

    async updateEyeColor(eyeColor) {
        this.config.eyeColor = eyeColor;
        console.log(`👁️ Updating eye color to: ${eyeColor}`);

        const eyeTexture = await this.loadEyeTexture(eyeColor);

        if (eyeTexture) {
            console.log(`✅ Applying eye texture for: ${eyeColor}`);
            this.applyEyeTexture(eyeTexture, eyeColor);
        } else {
            console.log(`⚠️ No texture available, using color: ${eyeColor}`);
            this.applyEyeColor(eyeColor);
        }
    }

    applyEyeTexture(texture, eyeColor) {
        let materialsUpdated = 0;

        this.avatarMaterials.eyes.forEach((material, index) => {
            if (material.map !== texture) {
                material.map = texture;
                material.needsUpdate = true;
                material.metalness = 0.0;
                material.roughness = 0.0;
                material.transparent = false;
                material.opacity = 1.0;

                if (material.isMeshPhongMaterial || material.isMeshLambertMaterial) {
                    material.shininess = 100;
                }

                materialsUpdated++;
                console.log(`  ✅ Applied eye texture to material ${index} (no roughness)`);
            }
        });

        console.log(`👁️ Updated ${materialsUpdated} materials with eye texture`);
    }

    applyEyeColor(eyeColor) {
        const newColor = this.eyeColors[eyeColor] || this.eyeColors['brown'];

        this.avatarMaterials.eyes.forEach((material, index) => {
            if (material.color) {
                material.color.setHex(newColor);
                material.metalness = 0.0;
                material.roughness = 0.0;
                material.needsUpdate = true;

                if (material.isMeshPhongMaterial || material.isMeshLambertMaterial) {
                    material.shininess = 100;
                }

                console.log(`  ✓ Updated eye material ${index} with color (no roughness)`);
            }
        });
    }

    async createTexturedFallbackEyes() {
        console.log('🔧 Creating textured fallback eyes...');

        if (!this.avatarModel) return;

        const eyeTexture = await this.loadEyeTexture(this.config.eyeColor);
        const eyeGeometry = new THREE.SphereGeometry(0.03, 16, 16);

        const eyeMaterial = new THREE.MeshPhongMaterial({
            map: eyeTexture,
            shininess: 100,
            transparent: false,
            metalness: 0.0,
            roughness: 0.0
        });

        if (!eyeTexture) {
            eyeMaterial.color.setHex(this.eyeColors[this.config.eyeColor] || this.eyeColors['brown']);
        }

        const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        leftEye.position.set(-0.035, 1.65, 0.08);
        leftEye.name = 'fallback_left_eye';

        const rightEyeMaterial = eyeMaterial.clone();
        const rightEye = new THREE.Mesh(eyeGeometry, rightEyeMaterial);
        rightEye.position.set(0.035, 1.65, 0.08);
        rightEye.name = 'fallback_right_eye';

        this.avatarModel.add(leftEye);
        this.avatarModel.add(rightEye);
        this.avatarMaterials.eyes.push(eyeMaterial, rightEyeMaterial);

        console.log('✅ Textured fallback eyes created (no roughness)');
    }

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

        this.avatarMaterials.skin.forEach((material, index) => {
            if (material.color) {
                material.color.setHex(newColor);
                material.needsUpdate = true;
                console.log(`  ✓ Updated skin material ${index}`);
            }
        });
    }

    updateHairColor(hairColor) {
        this.config.hairColor = hairColor;
        const newColor = this.hairColors[hairColor] || this.hairColors['brown'];

        console.log(`💇 Changing hair color to: ${hairColor} (0x${newColor.toString(16)})`);

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
        console.log('ℹ️ Note: Hair type changes in GLB require different model files');
    }

    getConfiguration() {
        return { ...this.config };
    }

    async setConfiguration(newConfig) {
        const oldConfig = { ...this.config };
        this.config = { ...this.config, ...newConfig };

        console.log('🔧 Setting new configuration:', newConfig);

        const majorChanges = ['gender', 'bodySize', 'height', 'hairType'];
        const needsReload = majorChanges.some(key =>
            oldConfig[key] !== newConfig[key] && newConfig[key] !== undefined
        );

        if (needsReload) {
            console.log('🔄 Major change detected, reloading avatar...');
            await this.loadAvatarFromConfig(this.config);
        } else {
            console.log('🎨 Minor change, updating materials...');
            if (oldConfig.skinColor !== newConfig.skinColor && newConfig.skinColor) {
                this.updateSkinColor(newConfig.skinColor);
            }
            if (oldConfig.eyeColor !== newConfig.eyeColor && newConfig.eyeColor) {
                await this.updateEyeColor(newConfig.eyeColor);
            }
            if (oldConfig.hairColor !== newConfig.hairColor && newConfig.hairColor) {
                this.updateHairColor(newConfig.hairColor);
            }
        }
    }

    checkEyeTextures() {
        console.log('🔍 Checking eye texture files...');

        Object.entries(this.eyeTextures).forEach(([color, path]) => {
            fetch(path)
                .then(response => {
                    if (response.ok) {
                        console.log(`✅ ${color}: ${path} - Available`);
                    } else {
                        console.warn(`❌ ${color}: ${path} - Missing (${response.status})`);
                    }
                })
                .catch(error => {
                    console.error(`❌ ${color}: ${path} - Error: ${error.message}`);
                });
        });
    }

    showExpectedTextures() {
        console.log('📋 EXPECTED EYE TEXTURE FILES:');
        console.log('Create these texture files for proper eye colors:');
        console.log('');
        console.log('📁 /static/models/makehuman/bodies/female/textures/');

        Object.entries(this.eyeTextures).forEach(([color, path]) => {
            const filename = path.split('/').pop();
            console.log(`   ├── ${filename} (for ${color} eyes)`);
        });

        console.log('');
        console.log('🎨 To create different colored eyes:');
        console.log('   1. Start with brown_eye.png as template');
        console.log('   2. Edit with image editor (GIMP, Photoshop, etc.)');
        console.log('   3. Change only the iris color, keep the pupil black');
        console.log('   4. Save with the appropriate filename');
        console.log('   5. Ensure all files are in the correct directory');
    }

    async forceApplyEyeTexture(eyeColor = null) {
        const colorToTest = eyeColor || this.config.eyeColor;

        if (!this.avatarModel) {
            console.error('❌ No avatar model loaded');
            return;
        }

        console.log(`🔧 Force applying ${colorToTest} eye texture for testing...`);

        const eyeTexture = await this.loadEyeTexture(colorToTest);

        if (eyeTexture) {
            let appliedCount = 0;

            this.avatarModel.traverse((child) => {
                if (child.isMesh && child.material) {
                    const meshName = (child.name || '').toLowerCase();
                    const materialName = (child.material.name || '').toLowerCase();

                    if (meshName.includes('head') || meshName.includes('face') ||
                        meshName.includes('eye') || materialName.includes('head') ||
                        materialName.includes('face') || materialName.includes('eye')) {

                        console.log(`🎯 Applying ${colorToTest} texture to: ${child.name} (${child.material.name})`);

                        const newMaterial = child.material.clone();
                        newMaterial.map = eyeTexture;
                        newMaterial.needsUpdate = true;
                        newMaterial.metalness = 0.0;
                        newMaterial.roughness = 0.0;
                        newMaterial.transparent = false;
                        newMaterial.opacity = 1.0;

                        if (newMaterial.isMeshPhongMaterial || newMaterial.isMeshLambertMaterial) {
                            newMaterial.shininess = 100;
                        }

                        child.material = newMaterial;
                        appliedCount++;
                    }
                }
            });

            console.log(`✅ Applied ${colorToTest} eye texture to ${appliedCount} materials`);
        } else {
            console.error(`❌ Failed to load ${colorToTest} eye texture`);
        }
    }

    createTestFallbackEyes() {
        if (!this.avatarModel) {
            console.error('❌ No avatar model loaded');
            return;
        }

        console.log('🔧 Creating test fallback eyes with texture...');

        const texturePath = this.eyeTextures[this.config.eyeColor];

        this.textureLoader.load(texturePath, (texture) => {
            texture.flipY = false;
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;

            const eyeGeometry = new THREE.SphereGeometry(0.04, 16, 16);
            const eyeMaterial = new THREE.MeshPhongMaterial({
                map: texture,
                transparent: false,
                metalness: 0.0,
                roughness: 0.0,
                shininess: 100
            });

            const existingTestEyes = this.avatarModel.children.filter(child =>
                child.name && child.name.includes('test_eye'));
            existingTestEyes.forEach(eye => this.avatarModel.remove(eye));

            const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
            leftEye.position.set(-0.04, 1.65, 0.09);
            leftEye.name = 'test_left_eye';

            const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial.clone());
            rightEye.position.set(0.04, 1.65, 0.09);
            rightEye.name = 'test_right_eye';

            this.avatarModel.add(leftEye);
            this.avatarModel.add(rightEye);

            console.log('✅ Test fallback eyes created with texture (no roughness)');
            console.log('👁️ Look at your avatar - you should see glossy textured eye spheres');
        }, undefined, (error) => {
            console.error('❌ Failed to load texture for test eyes:', error);
        });
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

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update();
        }

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    onWindowResize() {
        if (!this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);

        console.log(`📐 Resized to: ${width}x${height}`);
    }

    cleanup() {
        this.loadedTextures.forEach((texture) => {
            texture.dispose();
        });
        this.loadedTextures.clear();

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
window.MakeHumanAvatarManager = CustomizableGLBAvatarManager;

console.log('✅ Enhanced GLB Avatar Manager with Eye Textures loaded successfully');