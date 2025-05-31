/**
 * Enhanced MakeHuman Avatar Manager with proper OBJ/MTL rendering
 * Save this as: static/js/makehuman-avatar.js
 *
 * Features:
 * - Loads MakeHuman .obj files with .mtl materials and textures
 * - Proper scaling and positioning
 * - Gender/body size switching
 * - Real-time customization
 * - Fallback for missing files
 */

class MakeHumanAvatarManager {
    constructor(options = {}) {
        this.container = options.container || document.getElementById('avatar-container');
        this.debug = options.debug || false;
        this.enableStats = options.enableStats || true;

        // Avatar state
        this.avatarModel = null;
        this.currentAvatar = null;
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

        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Loaders
        this.objLoader = null;
        this.mtlLoader = null;
        this.textureLoader = null;

        // Initialize
        this.init();
    }

    init() {
        console.log('🤖 Initializing MakeHuman Avatar Manager...');
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLighting();
        this.setupControls();
        this.setupLoaders();
        this.setupGround();
        this.animate();

        // Load default avatar immediately
        this.loadDefaultAvatar();

        console.log('✅ MakeHuman Avatar Manager initialized');
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff); // White background

        // Add subtle fog for depth
        this.scene.fog = new THREE.Fog(0xffffff, 8, 20);
    }

    setupCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(0, 1.5, 3);
        this.camera.lookAt(0, 1, 0);
    }

    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Updated for newer Three.js versions and better color handling
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0; // Reduced exposure for better texture visibility

        // Ensure gamma correction for proper texture colors
        this.renderer.gammaFactor = 2.2;

        // Clear existing canvas and add new one
        const existingCanvas = this.container.querySelector('canvas');
        if (existingCanvas) {
            existingCanvas.remove();
        }
        this.container.appendChild(this.renderer.domElement);
    }

    setupLighting() {
        // Ambient light - increased for better visibility
        const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
        this.scene.add(ambientLight);

        // Key light (main light) - increased intensity
        const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
        keyLight.position.set(5, 10, 5);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.1;
        keyLight.shadow.camera.far = 50;
        keyLight.shadow.camera.left = -5;
        keyLight.shadow.camera.right = 5;
        keyLight.shadow.camera.top = 5;
        keyLight.shadow.camera.bottom = -5;
        this.scene.add(keyLight);

        // Fill light - increased intensity and better positioning
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);

        // Rim light - adjusted
        const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
        rimLight.position.set(-2, 3, -3);
        this.scene.add(rimLight);

        // Add front light for better texture visibility
        const frontLight = new THREE.DirectionalLight(0xffffff, 0.4);
        frontLight.position.set(0, 0, 10);
        this.scene.add(frontLight);
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
        } else {
            console.warn('⚠️ OrbitControls not available, using basic mouse controls');
            this.setupBasicControls();
        }
    }

    setupBasicControls() {
        let isMouseDown = false;
        let mouseX = 0, mouseY = 0;
        let cameraRadius = 3;
        let cameraTheta = 0;
        let cameraPhi = Math.PI / 3;

        this.renderer.domElement.addEventListener('mousedown', (e) => {
            isMouseDown = true;
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (!isMouseDown) return;

            const deltaX = e.clientX - mouseX;
            const deltaY = e.clientY - mouseY;

            cameraTheta -= deltaX * 0.01;
            cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, cameraPhi + deltaY * 0.01));

            this.updateCameraPosition(cameraRadius, cameraTheta, cameraPhi);

            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        this.renderer.domElement.addEventListener('mouseup', () => {
            isMouseDown = false;
        });

        this.renderer.domElement.addEventListener('wheel', (e) => {
            cameraRadius *= (e.deltaY > 0 ? 1.1 : 0.9);
            cameraRadius = Math.max(1, Math.min(10, cameraRadius));
            this.updateCameraPosition(cameraRadius, cameraTheta, cameraPhi);
        });
    }

    updateCameraPosition(radius, theta, phi) {
        const x = radius * Math.sin(phi) * Math.cos(theta);
        const y = radius * Math.cos(phi) + 1;
        const z = radius * Math.sin(phi) * Math.sin(theta);

        this.camera.position.set(x, y, z);
        this.camera.lookAt(0, 1, 0);
    }

    setupLoaders() {
        this.textureLoader = new THREE.TextureLoader();

        if (typeof THREE.MTLLoader !== 'undefined') {
            this.mtlLoader = new THREE.MTLLoader();
            // Override the texture loading to handle textures subfolder
            this.mtlLoader.setTexturePath = function(path) {
                this.texturePath = path;
                return this;
            };
        }

        if (typeof THREE.OBJLoader !== 'undefined') {
            this.objLoader = new THREE.OBJLoader();
        } else {
            console.error('❌ OBJLoader not available');
        }
    }

    setupGround() {
        const groundGeometry = new THREE.CircleGeometry(8, 32);
        const groundMaterial = new THREE.MeshLambertMaterial({
            color: 0xf5f5f5, // Light gray ground for white background
            transparent: true,
            opacity: 0.3
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    async loadDefaultAvatar() {
        console.log('🤖 Loading default MakeHuman avatar...');
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
        return this.loadMakeHumanAvatar(avatarPath);
    }

    getAvatarPath(config) {
        const { gender, bodySize, height } = config;

        // Default to m_medium for the starting point
        let filename = 'm_medium';

        // Map configuration to file paths based on your actual file structure
        const sizeMap = {
            'xs': 'xs',
            's': 's',
            'm': 'm',
            'l': 'l',
            'xl': 'xl'
        };

        const heightMap = {
            'short': 'short',
            'medium': 'medium',
            'tall': 'tall'
        };

        const actualSize = sizeMap[bodySize] || 'm';
        const actualHeight = heightMap[height] || 'medium';

        // Generate filename - always start with size_height pattern
        filename = `${actualSize}_${actualHeight}`;

        // Based on your file structure: static/models/makehuman/bodies/female/
        return `/static/models/makehuman/bodies/${gender}/${filename}`;
    }

    async loadMakeHumanAvatar(basePath) {
        console.log(`🔄 Loading MakeHuman avatar from: ${basePath}`);

        try {
            // Remove existing avatar
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
                this.avatarModel = null;
            }

            // Try multiple filename patterns to find the right file
            const possiblePaths = this.generatePossiblePaths(basePath);

            let loadedSuccessfully = false;
            for (const pathSet of possiblePaths) {
                try {
                    console.log(`🔍 Trying to load: ${pathSet.obj}`);

                    // Try to load materials first, then the OBJ
                    if (this.mtlLoader && pathSet.mtl) {
                        try {
                            const materials = await this.loadMTL(pathSet.mtl);
                            this.avatarModel = await this.loadOBJWithMTL(pathSet.obj, materials);
                        } catch (mtlError) {
                            console.warn('⚠️ MTL loading failed, trying OBJ only:', mtlError.message);
                            this.avatarModel = await this.loadOBJOnly(pathSet.obj);
                        }
                    } else {
                        this.avatarModel = await this.loadOBJOnly(pathSet.obj);
                    }

                    loadedSuccessfully = true;
                    console.log(`✅ Successfully loaded: ${pathSet.obj}`);
                    break;
                } catch (error) {
                    console.warn(`⚠️ Failed to load ${pathSet.obj}:`, error.message);
                    continue;
                }
            }

            if (!loadedSuccessfully) {
                throw new Error('No valid MakeHuman files found for this configuration');
            }

            // Configure the loaded avatar
            this.setupAvatarModel();

            // Force apply main skin texture if no textures were loaded
            this.forceApplyMainSkinTexture();

            this.scene.add(this.avatarModel);

            console.log('✅ MakeHuman avatar loaded successfully');
            return this.avatarModel;

        } catch (error) {
            console.error('❌ Error loading MakeHuman avatar:', error);
            throw error;
        }
    }

    generatePossiblePaths(basePath) {
        // Generate multiple possible file paths based on common MakeHuman naming patterns
        const baseDir = basePath.substring(0, basePath.lastIndexOf('/'));
        const filename = basePath.substring(basePath.lastIndexOf('/') + 1);

        const possibleFilenames = [
            filename,           // original: medium_medium
            'm_medium',         // your actual file
            'medium_m',         // alternative pattern
            'f_medium',         // female specific
            'male_medium',      // full gender name
            'female_medium',    // full gender name
            'medium',           // just size
            'default',          // fallback
            'base'              // another fallback
        ];

        return possibleFilenames.map(name => ({
            obj: `${baseDir}/${name}.obj`,
            mtl: `${baseDir}/${name}.mtl`
        }));
    }

    async loadMTL(mtlPath) {
        return new Promise((resolve, reject) => {
            console.log('🔄 Loading MTL from:', mtlPath);

            // First, let's read the MTL file content to see what it actually contains
            fetch(mtlPath)
                .then(response => response.text())
                .then(mtlContent => {
                    console.log('📄 MTL file content:');
                    console.log(mtlContent);

                    // Look for texture references in the MTL content
                    const textureReferences = mtlContent.match(/map_Kd\s+(.+)/g);
                    if (textureReferences) {
                        console.log('🎨 Texture references found in MTL:', textureReferences);
                    } else {
                        console.log('⚠️ No texture references found in MTL file');
                    }
                })
                .catch(err => console.warn('Could not read MTL content:', err));

            // Set the texture path before loading - use relative path to avoid duplication
            const basePath = mtlPath.substring(0, mtlPath.lastIndexOf('/') + 1);
            const texturesPath = basePath + 'textures/';

            // Clear any existing path to avoid duplication
            this.mtlLoader.setPath('');

            this.mtlLoader.load(
                mtlPath,
                (materials) => {
                    materials.preload();
                    console.log('✅ MTL materials loaded from:', mtlPath);
                    console.log('📋 Available materials:', Object.keys(materials.materials));

                    // Debug each material in detail
                    Object.entries(materials.materials).forEach(([name, material], index) => {
                        console.log(`🔍 MTL Material ${index + 1} - ${name}:`, {
                            type: material.type,
                            color: material.color?.getHexString(),
                            hasMap: !!material.map,
                            mapUrl: material.map?.image?.src,
                            mapImage: material.map?.image,
                            hasNormalMap: !!material.normalMap,
                            hasSpecularMap: !!material.specularMap,
                            opacity: material.opacity,
                            transparent: material.transparent,
                            side: material.side
                        });

                        // If material has a map but image failed to load
                        if (material.map && (!material.map.image || material.map.image.width === 0)) {
                            console.warn('⚠️ MTL material has map but image failed to load:', material.map.image?.src);

                            // Try to manually load the main skin texture for this material
                            const mainSkinTexture = texturesPath + 'young_lightskinned_female_diffuse.png';
                            console.log('🔄 Manually loading skin texture for failed material:', mainSkinTexture);

                            material.map = this.textureLoader.load(
                                mainSkinTexture,
                                (tex) => {
                                    console.log('✅ Manually loaded skin texture for material:', name);
                                    tex.needsUpdate = true;
                                    tex.wrapS = THREE.RepeatWrapping;
                                    tex.wrapT = THREE.RepeatWrapping;
                                    material.needsUpdate = true;
                                },
                                undefined,
                                (error) => {
                                    console.warn('⚠️ Failed to manually load skin texture:', error);
                                }
                            );
                        } else if (!material.map) {
                            // No texture map at all, force load the main skin texture
                            console.log('🎨 No texture map found in MTL, force loading main skin texture for:', name);
                            const mainSkinTexture = texturesPath + 'young_lightskinned_female_diffuse.png';
                            console.log('🔄 Force loading skin texture:', mainSkinTexture);

                            material.map = this.textureLoader.load(
                                mainSkinTexture,
                                (tex) => {
                                    console.log('✅ Force loaded main skin texture for MTL material:', name);
                                    tex.needsUpdate = true;
                                    tex.wrapS = THREE.RepeatWrapping;
                                    tex.wrapT = THREE.RepeatWrapping;
                                    material.needsUpdate = true;
                                },
                                undefined,
                                (error) => {
                                    console.log('ℹ️ Could not force load skin texture, using color fallback for:', name);
                                    material.color.setHex(this.getSkinColor(this.config.skinColor));
                                }
                            );
                        } else {
                            console.log('✅ MTL material already has valid texture map:', name);
                        }

                        // Ensure material is properly configured
                        material.side = THREE.DoubleSide;
                        material.needsUpdate = true;

                        // Enhance material properties for better rendering
                        if (material.map) {
                            material.map.needsUpdate = true;
                            material.map.wrapS = THREE.RepeatWrapping;
                            material.map.wrapT = THREE.RepeatWrapping;
                            console.log('✅ Texture map configured for MTL material:', material.map.image?.src || 'loading...');
                        }

                        if (material.normalMap) {
                            material.normalMap.needsUpdate = true;
                        }

                        if (material.specularMap) {
                            material.specularMap.needsUpdate = true;
                        }

                        // Ensure proper lighting response
                        if (material.isMeshPhongMaterial || material.isMeshLambertMaterial) {
                            material.shininess = material.shininess || 30;
                        }
                    });

                    resolve(materials);
                },
                (progress) => {
                    console.log('📊 MTL loading progress:', progress);
                },
                (error) => {
                    console.warn('⚠️ Failed to load MTL:', mtlPath, error);
                    resolve(null);
                }
            );
        });
    }

    async loadOBJWithMTL(objPath, materials) {
        return new Promise((resolve, reject) => {
            if (materials) {
                this.objLoader.setMaterials(materials);
                console.log('🎨 Setting materials for OBJ loading:', Object.keys(materials.materials));
            }

            this.objLoader.load(
                objPath,
                (object) => {
                    // Ensure materials are properly applied
                    if (materials) {
                        object.traverse((child) => {
                            if (child.isMesh) {
                                console.log('🔍 Processing mesh:', child.name, 'Current material:', child.material?.name || 'unnamed');

                                // Find and apply the appropriate material
                                const materialName = child.material?.name;
                                if (materialName && materials.materials[materialName]) {
                                    child.material = materials.materials[materialName];
                                    console.log('✅ Applied material:', materialName);
                                } else {
                                    // Try to find any skin-related material
                                    const skinMaterial = Object.values(materials.materials).find(mat =>
                                        mat.name && (mat.name.includes('skin') || mat.name.includes('body') || mat.name.includes('flesh'))
                                    );

                                    if (skinMaterial) {
                                        child.material = skinMaterial;
                                        console.log('✅ Applied skin material:', skinMaterial.name);
                                    } else {
                                        // Use the first available material
                                        const firstMaterial = Object.values(materials.materials)[0];
                                        if (firstMaterial) {
                                            child.material = firstMaterial;
                                            console.log('✅ Applied first available material:', firstMaterial.name);
                                        }
                                    }
                                }

                                // Ensure material properties are set correctly
                                if (child.material) {
                                    child.material.needsUpdate = true;
                                    child.material.side = THREE.DoubleSide;

                                    // Debug material properties
                                    console.log('🔍 Material properties:', {
                                        name: child.material.name,
                                        hasTexture: !!child.material.map,
                                        color: child.material.color?.getHexString(),
                                        textureUrl: child.material.map?.image?.src
                                    });

                                    // If material has textures, ensure they're loaded
                                    if (child.material.map) {
                                        child.material.map.needsUpdate = true;
                                    }
                                    if (child.material.normalMap) {
                                        child.material.normalMap.needsUpdate = true;
                                    }
                                }
                            }
                        });
                    }

                    console.log('✅ OBJ with MTL loaded');
                    resolve(object);
                },
                (progress) => {
                    console.log('📊 OBJ loading progress:', progress);
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }

    async loadOBJOnly(objPath) {
        return new Promise((resolve, reject) => {
            this.objLoader.load(
                objPath,
                (object) => {
                    // Apply default skin material with proper texture loading attempt
                    const basePath = objPath.substring(0, objPath.lastIndexOf('/') + 1);

                    object.traverse((child) => {
                        if (child.isMesh) {
                            console.log('🎨 Applying materials to mesh:', child.name || 'unnamed');

                            // Try texture files based on your actual file structure
                            const possibleTextures = [
                                // Your specific texture files in textures subfolder
                                'textures/young_lightskinned_female_diffuse.png',
                                'textures/fr_thong.png',
                                'textures/frenchbra.png',
                                'textures/brown_eye.png',
                                // Fallback texture names
                                'textures/texture.jpg',
                                'textures/diffuse.jpg',
                                'textures/skin.jpg',
                                'textures/texture.png',
                                'textures/diffuse.png',
                                // Base folder fallbacks
                                'young_lightskinned_female_diffuse.png',
                                'texture.jpg',
                                'diffuse.jpg',
                                'skin.jpg',
                                'texture.png',
                                'diffuse.png'
                            ];

                            let textureLoaded = false;

                            for (const textureName of possibleTextures) {
                                try {
                                    const fullTexturePath = basePath + textureName;
                                    console.log('🔍 Trying texture:', fullTexturePath);

                                    const texture = this.textureLoader.load(
                                        fullTexturePath,
                                        (tex) => {
                                            console.log('✅ Texture loaded successfully:', fullTexturePath);
                                            tex.needsUpdate = true;
                                            tex.wrapS = THREE.RepeatWrapping;
                                            tex.wrapT = THREE.RepeatWrapping;
                                            textureLoaded = true;

                                            // Update material with texture
                                            child.material = new THREE.MeshPhongMaterial({
                                                map: tex,
                                                color: 0xffffff, // White to show full texture
                                                side: THREE.DoubleSide,
                                                shininess: 30
                                            });
                                        },
                                        undefined,
                                        (error) => {
                                            console.log(`ℹ️ Texture not found: ${textureName}`);
                                        }
                                    );

                                    if (textureLoaded) break;
                                } catch (error) {
                                    console.log(`ℹ️ Failed to load texture: ${textureName}`);
                                }
                            }

                            // If no texture found, use color-based material
                            if (!textureLoaded) {
                                console.log('🎨 No texture found, using default skin color material');
                                child.material = new THREE.MeshPhongMaterial({
                                    color: this.getSkinColor(this.config.skinColor),
                                    side: THREE.DoubleSide,
                                    shininess: 30
                                });
                            }
                        }
                    });
                    console.log('✅ OBJ loaded with materials applied');
                    resolve(object);
                },
                (progress) => {
                    console.log('📊 OBJ loading progress:', progress);
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }

    createDefaultSkinMaterial() {
        return new THREE.MeshLambertMaterial({
            color: this.getSkinColor(this.config.skinColor),
            side: THREE.DoubleSide
        });
    }

    getSkinColor(skinType) {
        const skinColors = {
            'light': 0xfdbcb4,
            'medium': 0xee9b82,
            'tan': 0xd08b5b,
            'dark': 0xae5d29,
            'darker': 0x8b4513,
            'darkest': 0x654321,
            'pink': 0xffb6c1,
            'olive': 0xddbea9
        };
        return skinColors[skinType] || skinColors['light'];
    }

    setupAvatarModel() {
        if (!this.avatarModel) return;

        // Calculate model bounds for proper scaling and positioning
        const box = new THREE.Box3().setFromObject(this.avatarModel);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        // Auto-scale to appropriate size (assuming MakeHuman models are in meters)
        let scale = 1;
        if (size.y > 10) {
            scale = 1.8 / size.y; // Scale to ~1.8m height
        } else if (size.y < 0.1) {
            scale = 1.8 / size.y; // Scale up if too small
        }

        this.avatarModel.scale.setScalar(scale);

        // Position at ground level
        this.avatarModel.position.y = -box.min.y * scale;
        this.avatarModel.position.x = -center.x * scale;
        this.avatarModel.position.z = -center.z * scale;

        // Enable shadows and debug materials
        this.avatarModel.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;

                // Debug current material state
                console.log('🔍 Mesh material debug:', {
                    meshName: child.name || 'unnamed',
                    materialType: child.material?.type,
                    materialName: child.material?.name,
                    hasTexture: !!child.material?.map,
                    textureLoaded: child.material?.map?.image ? 'loaded' : 'not loaded',
                    materialColor: child.material?.color?.getHexString(),
                    textureUrl: child.material?.map?.image?.src
                });

                // Ensure materials are properly configured
                if (child.material) {
                    child.material.needsUpdate = true;
                }
            }
        });

        console.log(`📏 Avatar scaled by ${scale.toFixed(2)}, positioned at (${this.avatarModel.position.x.toFixed(2)}, ${this.avatarModel.position.y.toFixed(2)}, ${this.avatarModel.position.z.toFixed(2)})`);

        // Debug the overall model
        this.debugAvatarMaterials();
    }

    debugAvatarMaterials() {
        if (!this.avatarModel) return;

        console.log('🔍 === AVATAR MATERIAL DEBUG ===');
        let meshCount = 0;
        let texturedMeshes = 0;
        let untexturedMeshes = 0;

        this.avatarModel.traverse((child) => {
            if (child.isMesh) {
                meshCount++;
                const hasTexture = child.material?.map && child.material.map.image;

                if (hasTexture) {
                    texturedMeshes++;
                    console.log(`✅ Textured mesh: ${child.name || 'unnamed'} - ${child.material.map.image.src}`);
                } else {
                    untexturedMeshes++;
                    console.log(`❌ Untextured mesh: ${child.name || 'unnamed'} - Color: ${child.material?.color?.getHexString() || 'none'}`);
                }
            }
        });

        console.log(`📊 Material Summary: ${meshCount} total meshes, ${texturedMeshes} textured, ${untexturedMeshes} untextured`);
        console.log('🔍 === END MATERIAL DEBUG ===');
    }

    forceApplyMainSkinTexture() {
        if (!this.avatarModel) return;

        console.log('🎨 Force applying main skin texture...');

        // Get the base path from the current avatar path
        const avatarPath = this.getAvatarPath(this.config);
        const basePath = avatarPath.substring(0, avatarPath.lastIndexOf('/') + 1);
        const mainSkinTexture = basePath + 'textures/young_lightskinned_female_diffuse.png';

        console.log('🔍 Trying to force load:', mainSkinTexture);

        // Always try to apply texture, regardless of current state
        console.log('🔄 Force loading main skin texture...');

        this.textureLoader.load(
            mainSkinTexture,
            (texture) => {
                console.log('✅ Main skin texture loaded successfully, applying to ALL meshes');
                texture.needsUpdate = true;
                texture.wrapS = THREE.RepeatWrapping;
                texture.wrapT = THREE.RepeatWrapping;
                texture.flipY = false; // Important for some OBJ textures

                // Force apply to ALL meshes
                this.avatarModel.traverse((child) => {
                    if (child.isMesh) {
                        console.log('🎨 Force applying texture to mesh:', child.name || 'unnamed');

                        // Create a new material with enhanced settings for better visibility
                        const newMaterial = new THREE.MeshLambertMaterial({
                            map: texture,
                            color: 0xffffff, // White to show full texture
                            side: THREE.DoubleSide,
                            transparent: false,
                            alphaTest: 0.1,
                            // Remove shininess for Lambert material
                        });

                        // Replace the material completely
                        if (child.material) {
                            // Dispose old material
                            if (child.material.dispose) {
                                child.material.dispose();
                            }
                        }

                        child.material = newMaterial;
                        child.material.needsUpdate = true;

                        console.log('✅ Applied new textured material to:', child.name || 'unnamed');
                    }
                });

                // Force a render update
                if (this.renderer) {
                    this.renderer.render(this.scene, this.camera);
                }
            },
            (progress) => {
                console.log('📊 Texture loading progress:', progress);
            },
            (error) => {
                console.warn('⚠️ Failed to force load main skin texture:', error);
                console.log('🎨 Using enhanced color-based materials as final fallback');

                // Enhanced fallback: apply better skin color to all meshes
                this.avatarModel.traverse((child) => {
                    if (child.isMesh) {
                        console.log('🎨 Applying color fallback to:', child.name || 'unnamed');

                        const colorMaterial = new THREE.MeshLambertMaterial({
                            color: this.getSkinColor(this.config.skinColor),
                            side: THREE.DoubleSide,
                            transparent: false
                        });

                        if (child.material && child.material.dispose) {
                            child.material.dispose();
                        }

                        child.material = colorMaterial;
                        child.material.needsUpdate = true;
                    }
                });

                // Force a render update
                if (this.renderer) {
                    this.renderer.render(this.scene, this.camera);
                }
            }
        );
    }

    loadFallbackAvatar() {
        console.log('🔄 Loading fallback avatar...');

        // Create a simple fallback avatar
        const geometry = new THREE.CapsuleGeometry(0.3, 1.4, 4, 8);
        const material = new THREE.MeshLambertMaterial({
            color: this.getSkinColor(this.config.skinColor)
        });

        this.avatarModel = new THREE.Mesh(geometry, material);
        this.avatarModel.position.y = 0.9;
        this.avatarModel.castShadow = true;
        this.avatarModel.receiveShadow = true;

        this.scene.add(this.avatarModel);
        console.log('✅ Fallback avatar created');
    }

    // Avatar customization methods
    async updateGender(gender) {
        this.config.gender = gender;
        await this.loadAvatarFromConfig(this.config);
    }

    async updateBodySize(bodySize) {
        this.config.bodySize = bodySize;
        await this.loadAvatarFromConfig(this.config);
    }

    async updateHeight(height) {
        this.config.height = height;
        await this.loadAvatarFromConfig(this.config);
    }

    updateSkinColor(skinColor) {
        this.config.skinColor = skinColor;

        if (this.avatarModel) {
            const newColor = this.getSkinColor(skinColor);
            this.avatarModel.traverse((child) => {
                if (child.isMesh && child.material) {
                    if (child.material.name && child.material.name.includes('skin')) {
                        child.material.color.setHex(newColor);
                    } else if (!child.material.name) {
                        // If no material name, assume it's skin
                        child.material.color.setHex(newColor);
                    }
                }
            });
        }
    }

    updateEyeColor(eyeColor) {
        this.config.eyeColor = eyeColor;
        // Eye color changes would require more complex material manipulation
        console.log(`👁️ Eye color set to: ${eyeColor}`);
    }

    async updateHairType(hairType) {
        this.config.hairType = hairType;
        console.log(`✂️ Hair type set to: ${hairType}`);
        // Hair would be a separate model/accessory to load
    }

    updateHairColor(hairColor) {
        this.config.hairColor = hairColor;
        console.log(`🎨 Hair color set to: ${hairColor}`);
        // Hair color changes would require hair model to be loaded first
    }

    // Utility methods
    getConfiguration() {
        return { ...this.config };
    }

    async setConfiguration(newConfig) {
        this.config = { ...this.config, ...newConfig };
        await this.loadAvatarFromConfig(this.config);
    }

    clearAllClothing() {
        this.clothingItems.forEach((item, id) => {
            this.removeClothing(id);
        });
        this.clothingItems.clear();
    }

    removeClothing(itemId) {
        const item = this.clothingItems.get(itemId);
        if (item && item.model) {
            this.scene.remove(item.model);
            this.clothingItems.delete(itemId);
            return true;
        }
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

    // Resize handler
    onWindowResize() {
        if (!this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    // Cleanup
    cleanup() {
        if (this.renderer) {
            this.renderer.dispose();
        }

        if (this.scene) {
            // Dispose of all geometries and materials
            this.scene.traverse((object) => {
                if (object.geometry) {
                    object.geometry.dispose();
                }
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(material => material.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        }

        console.log('🧹 MakeHuman Avatar Manager cleaned up');
    }
}

// Make globally available
window.MakeHumanAvatarManager = MakeHumanAvatarManager;