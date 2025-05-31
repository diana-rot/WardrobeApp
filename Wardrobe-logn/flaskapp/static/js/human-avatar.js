// human-avatar.js - Enhanced Avatar System with Clothing Controls
// Place this file at: flaskapp/static/js/human-avatar.js

class EnhancedAvatarWardrobeSystem {
    constructor() {
        console.log('🚀 Initializing Enhanced Avatar Wardrobe System...');

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.avatar = null;
        this.currentClothing = new Map();

        // Clothing control system
        this.selectedClothing = null;
        this.clothingControls = null;
        this.isControlMode = false;
        this.transformControls = null;

        // Loaders
        this.objLoader = null;
        this.mtlLoader = null;
        this.gltfLoader = null;
        this.textureLoader = null;

        // UI elements
        this.clothingControlsPanel = null;
        this.resetButton = null;
        this.saveButton = null;

        this.initializeSystem();
    }

    async initializeSystem() {
        console.log('🔧 Setting up Enhanced Avatar System...');

        // Wait for THREE.js to be ready
        if (typeof THREE === 'undefined') {
            console.log('⏳ Waiting for THREE.js...');
            await this.waitForThreeJS();
        }

        this.initializeLoaders();
        this.setupScene();
        this.setupClothingControls();
        this.setupUI();

        // Make globally available
        window.enhancedAvatarSystem = this;

        console.log('✅ Enhanced Avatar Wardrobe System initialized successfully!');
    }

    waitForThreeJS(maxAttempts = 50) {
        return new Promise((resolve, reject) => {
            let attempts = 0;

            const checkThree = () => {
                attempts++;

                if (typeof THREE !== 'undefined' && THREE.OBJLoader && THREE.GLTFLoader) {
                    console.log('✅ THREE.js is ready!');
                    resolve(true);
                } else if (attempts < maxAttempts) {
                    console.log(`⏳ Waiting for THREE.js... (${attempts}/${maxAttempts})`);
                    setTimeout(checkThree, 200);
                } else {
                    console.error('❌ THREE.js failed to load after maximum attempts');
                    reject(new Error('THREE.js failed to load'));
                }
            };

            checkThree();
        });
    }

    initializeLoaders() {
        console.log('🔄 Initializing loaders...');

        try {
            this.objLoader = new THREE.OBJLoader();
            this.gltfLoader = new THREE.GLTFLoader();
            this.textureLoader = new THREE.TextureLoader();

            if (THREE.MTLLoader) {
                this.mtlLoader = new THREE.MTLLoader();
            }

            console.log('✅ All loaders initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing loaders:', error);
        }
    }

    setupScene() {
        console.log('🎬 Setting up 3D scene...');

        const container = document.getElementById('avatar-container');
        if (!container) {
            console.error('❌ Avatar container not found');
            return;
        }

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf0f0f0);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            50,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 1.6, 4);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 1, 0);
        this.controls.update();

        // Lighting
        this.setupLighting();

        // Transform controls for clothing manipulation
        this.transformControls = new THREE.TransformControls(this.camera, this.renderer.domElement);
        this.scene.add(this.transformControls);

        // Handle transform controls events
        this.transformControls.addEventListener('change', () => {
            this.renderer.render(this.scene, this.camera);
        });

        this.transformControls.addEventListener('dragging-changed', (event) => {
            this.controls.enabled = !event.value;
        });

        // Start render loop
        this.animate();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        console.log('✅ 3D scene setup complete');
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Main directional light
        const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
        mainLight.position.set(5, 10, 5);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.width = 2048;
        mainLight.shadow.mapSize.height = 2048;
        this.scene.add(mainLight);

        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);

        // Ground
        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.MeshLambertMaterial({ color: 0xcccccc });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    setupClothingControls() {
        console.log('🎮 Setting up clothing controls...');

        // Add mouse interaction for clothing selection
        this.setupMouseInteraction();

        console.log('✅ Clothing controls setup complete');
    }

    setupMouseInteraction() {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        this.renderer.domElement.addEventListener('click', (event) => {
            if (this.isControlMode) return;

            const rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, this.camera);

            // Check for clothing intersections
            const clothingMeshes = Array.from(this.currentClothing.values()).map(item => item.mesh);
            const intersects = raycaster.intersectObjects(clothingMeshes, true);

            if (intersects.length > 0) {
                const selectedMesh = intersects[0].object;
                this.selectClothing(selectedMesh);
            } else {
                this.deselectClothing();
            }
        });
    }

    selectClothing(mesh) {
        console.log('👕 Selecting clothing item:', mesh);

        // Find the clothing item
        let selectedItem = null;
        for (const [itemId, clothingData] of this.currentClothing.entries()) {
            if (clothingData.mesh === mesh || clothingData.mesh.children.includes(mesh)) {
                selectedItem = { itemId, ...clothingData };
                break;
            }
        }

        if (selectedItem) {
            this.selectedClothing = selectedItem;
            this.showClothingControls(selectedItem);
            this.highlightClothing(selectedItem.mesh);
        }
    }

    deselectClothing() {
        if (this.selectedClothing) {
            this.removeHighlight(this.selectedClothing.mesh);
            this.selectedClothing = null;
            this.hideClothingControls();
            this.transformControls.detach();
        }
    }

    highlightClothing(mesh) {
        // Add selection outline or highlight
        if (mesh.userData.originalMaterial) return; // Already highlighted

        mesh.traverse((child) => {
            if (child.isMesh && child.material) {
                child.userData.originalMaterial = child.material.clone();
                child.material = child.material.clone();
                child.material.emissive = new THREE.Color(0x444444);
            }
        });
    }

    removeHighlight(mesh) {
        mesh.traverse((child) => {
            if (child.isMesh && child.userData.originalMaterial) {
                child.material = child.userData.originalMaterial;
                delete child.userData.originalMaterial;
            }
        });
    }

    showClothingControls(clothingItem) {
        console.log('🎛️ Showing clothing controls for:', clothingItem.itemData.label);

        // Attach transform controls to the selected clothing
        this.transformControls.attach(clothingItem.mesh);
        this.isControlMode = true;

        // Show UI controls
        this.createClothingControlsUI(clothingItem);
    }

    hideClothingControls() {
        this.isControlMode = false;
        this.removeClothingControlsUI();
    }

    createClothingControlsUI(clothingItem) {
        // Remove existing controls
        this.removeClothingControlsUI();

        // Create controls panel
        this.clothingControlsPanel = document.createElement('div');
        this.clothingControlsPanel.className = 'clothing-controls-panel';
        this.clothingControlsPanel.innerHTML = `
            <div class="clothing-controls-header">
                <h4>👕 ${clothingItem.itemData.label}</h4>
                <button class="close-controls" onclick="enhancedAvatarSystem.deselectClothing()">×</button>
            </div>
            <div class="clothing-controls-body">
                <div class="control-group">
                    <label>Transform Mode:</label>
                    <button class="control-btn active" data-mode="translate">Move</button>
                    <button class="control-btn" data-mode="rotate">Rotate</button>
                    <button class="control-btn" data-mode="scale">Scale</button>
                </div>
                <div class="control-group">
                    <label>Quick Actions:</label>
                    <button class="control-btn" onclick="enhancedAvatarSystem.resetClothingPosition('${clothingItem.itemId}')">Reset Position</button>
                    <button class="control-btn" onclick="enhancedAvatarSystem.fitToAvatar('${clothingItem.itemId}')">Fit to Avatar</button>
                </div>
                <div class="control-group">
                    <label>Position:</label>
                    <div class="position-controls">
                        <input type="number" id="pos-x" step="0.01" placeholder="X">
                        <input type="number" id="pos-y" step="0.01" placeholder="Y">
                        <input type="number" id="pos-z" step="0.01" placeholder="Z">
                    </div>
                </div>
                <div class="control-group">
                    <button class="control-btn save-btn" onclick="enhancedAvatarSystem.saveClothingTransform('${clothingItem.itemId}')">💾 Save Changes</button>
                </div>
            </div>
        `;

        // Add styles
        this.addControlsCSS();

        // Add to page
        document.body.appendChild(this.clothingControlsPanel);

        // Setup event listeners
        this.setupControlsEvents(clothingItem);

        // Update position inputs with current values
        this.updatePositionInputs(clothingItem.mesh);
    }

    setupControlsEvents(clothingItem) {
        // Transform mode buttons
        const modeButtons = this.clothingControlsPanel.querySelectorAll('[data-mode]');
        modeButtons.forEach(button => {
            button.addEventListener('click', () => {
                modeButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                this.transformControls.setMode(button.dataset.mode);
            });
        });

        // Position inputs
        const posInputs = this.clothingControlsPanel.querySelectorAll('.position-controls input');
        posInputs.forEach((input, index) => {
            input.addEventListener('change', () => {
                const axes = ['x', 'y', 'z'];
                const value = parseFloat(input.value) || 0;
                clothingItem.mesh.position[axes[index]] = value;
                this.renderer.render(this.scene, this.camera);
            });
        });

        // Update position inputs when transform controls change
        this.transformControls.addEventListener('objectChange', () => {
            this.updatePositionInputs(clothingItem.mesh);
        });
    }

    updatePositionInputs(mesh) {
        if (!this.clothingControlsPanel) return;

        const posInputs = this.clothingControlsPanel.querySelectorAll('.position-controls input');
        if (posInputs.length >= 3) {
            posInputs[0].value = mesh.position.x.toFixed(2);
            posInputs[1].value = mesh.position.y.toFixed(2);
            posInputs[2].value = mesh.position.z.toFixed(2);
        }
    }

    removeClothingControlsUI() {
        if (this.clothingControlsPanel) {
            this.clothingControlsPanel.remove();
            this.clothingControlsPanel = null;
        }
    }

    addControlsCSS() {
        if (document.getElementById('clothing-controls-css')) return;

        const style = document.createElement('style');
        style.id = 'clothing-controls-css';
        style.textContent = `
            .clothing-controls-panel {
                position: fixed;
                top: 20px;
                right: 20px;
                width: 280px;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                backdrop-filter: blur(10px);
                z-index: 1000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }

            .clothing-controls-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px;
                border-bottom: 1px solid rgba(0,0,0,0.1);
            }

            .clothing-controls-header h4 {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
                color: #333;
            }

            .close-controls {
                background: none;
                border: none;
                font-size: 20px;
                color: #666;
                cursor: pointer;
                padding: 4px;
                border-radius: 4px;
            }

            .close-controls:hover {
                background: rgba(0,0,0,0.1);
            }

            .clothing-controls-body {
                padding: 16px;
            }

            .control-group {
                margin-bottom: 16px;
            }

            .control-group label {
                display: block;
                font-size: 12px;
                font-weight: 600;
                color: #666;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .control-btn {
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                margin-right: 6px;
                margin-bottom: 6px;
                transition: all 0.2s ease;
            }

            .control-btn:hover {
                background: #e9e9e9;
                border-color: #bbb;
            }

            .control-btn.active {
                background: #3498db;
                color: white;
                border-color: #3498db;
            }

            .control-btn.save-btn {
                background: #27ae60;
                color: white;
                border-color: #27ae60;
                width: 100%;
                margin-right: 0;
            }

            .control-btn.save-btn:hover {
                background: #229954;
            }

            .position-controls {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 6px;
            }

            .position-controls input {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 12px;
                text-align: center;
            }

            .position-controls input:focus {
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }
        `;
        document.head.appendChild(style);
    }

    setupUI() {
        console.log('🖥️ Setting up user interface...');

        // Add clothing management buttons to the wardrobe tab
        this.addWardrobeButtons();
    }

    addWardrobeButtons() {
        const wardrobeTab = document.getElementById('wardrobe');
        if (!wardrobeTab) return;

        // Add avatar mode toggle
        const avatarControls = document.createElement('div');
        avatarControls.className = 'avatar-mode-controls';
        avatarControls.innerHTML = `
            <div class="form-group">
                <label>👕 Clothing Mode:</label>
                <div class="mode-toggle">
                    <button class="mode-btn active" data-mode="view">View Mode</button>
                    <button class="mode-btn" data-mode="control">Control Mode</button>
                </div>
            </div>
            <div class="form-group">
                <button id="clear-all-clothing" class="btn btn-secondary">
                    <i class="bi bi-x-circle"></i> Clear All Clothing
                </button>
                <button id="reset-clothing-positions" class="btn btn-info">
                    <i class="bi bi-arrow-counterclockwise"></i> Reset Positions
                </button>
            </div>
        `;

        // Insert after the wardrobe section title
        const wardrobeSection = wardrobeTab.querySelector('.wardrobe-section');
        if (wardrobeSection) {
            wardrobeSection.insertBefore(avatarControls, wardrobeSection.children[2]);
        }

        // Add event listeners
        this.setupWardrobeEvents();
    }

    setupWardrobeEvents() {
        // Mode toggle
        document.addEventListener('click', (e) => {
            if (e.target.matches('.mode-btn')) {
                document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');

                const mode = e.target.dataset.mode;
                this.setInteractionMode(mode);
            }
        });

        // Clear all clothing
        document.addEventListener('click', (e) => {
            if (e.target.matches('#clear-all-clothing')) {
                this.clearAllClothing();
            }
        });

        // Reset clothing positions
        document.addEventListener('click', (e) => {
            if (e.target.matches('#reset-clothing-positions')) {
                this.resetAllClothingPositions();
            }
        });
    }

    setInteractionMode(mode) {
        console.log(`🎮 Setting interaction mode: ${mode}`);

        if (mode === 'control') {
            this.controls.enabled = false;
            this.isControlMode = true;
            this.showMessage('Control mode enabled - click clothing to adjust', 'info');
        } else {
            this.controls.enabled = true;
            this.isControlMode = false;
            this.deselectClothing();
            this.showMessage('View mode enabled - drag to rotate camera', 'info');
        }
    }

    // Avatar and Clothing Management Methods
    async loadAvatar(avatarPath) {
        console.log('👤 Loading avatar:', avatarPath);

        try {
            if (this.avatar) {
                this.scene.remove(this.avatar);
            }

            const gltf = await this.loadGLTF(avatarPath);
            this.avatar = gltf.scene;

            // Configure avatar
            this.avatar.scale.setScalar(1);
            this.avatar.position.set(0, 0, 0);

            // Enable shadows
            this.avatar.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });

            this.scene.add(this.avatar);
            console.log('✅ Avatar loaded successfully');

            return true;
        } catch (error) {
            console.error('❌ Error loading avatar:', error);
            return false;
        }
    }

    async loadClothingItem(itemId) {
        console.log(`👕 Loading clothing item: ${itemId}`);

        try {
            // Get item data from API
            const response = await fetch(`/api/wardrobe/item/${itemId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const itemData = await response.json();
            if (!itemData.success) {
                throw new Error(itemData.error || 'Failed to fetch item data');
            }

            // Try to load 3D model or create from texture
            const clothingMesh = await this.createClothingMesh(itemData);

            if (clothingMesh) {
                // Position clothing on avatar
                this.positionClothingOnAvatar(clothingMesh, itemData);

                // Add to scene
                this.scene.add(clothingMesh);

                // Store clothing data
                this.currentClothing.set(itemId, {
                    mesh: clothingMesh,
                    itemData: itemData,
                    originalPosition: clothingMesh.position.clone(),
                    originalRotation: clothingMesh.rotation.clone(),
                    originalScale: clothingMesh.scale.clone()
                });

                console.log(`✅ Clothing item ${itemId} loaded successfully`);
                this.showMessage(`Added ${itemData.label} to avatar`, 'success');
                return true;
            }

        } catch (error) {
            console.error(`❌ Error loading clothing item ${itemId}:`, error);
            this.showMessage(`Failed to load clothing item: ${error.message}`, 'error');
            return false;
        }
    }

    async createClothingMesh(itemData) {
        // Try to load 3D model first
        if (itemData.has_3d_model && itemData.model_3d_path) {
            try {
                const objMesh = await this.loadOBJFile(itemData.model_3d_path);
                this.applyColorToMesh(objMesh, itemData.color);
                return objMesh;
            } catch (error) {
                console.warn('⚠️ Failed to load 3D model, creating textured plane:', error);
            }
        }

        // Create textured plane as fallback
        return this.createTexturedPlane(itemData);
    }

    async loadOBJFile(objPath) {
        return new Promise((resolve, reject) => {
            this.objLoader.load(
                objPath,
                (object) => {
                    // Apply proper rotation for clothing
                    object.rotation.set(-Math.PI / 2, 0, Math.PI / 2);
                    resolve(object);
                },
                (progress) => {
                    console.log('📥 Loading OBJ:', Math.round(progress.loaded / progress.total * 100), '%');
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }

    createTexturedPlane(itemData) {
        const geometry = this.getGeometryForClothingType(itemData.type || itemData.label);

        // Create material with texture if available
        let material;
        if (itemData.file_path) {
            const texture = this.textureLoader.load(itemData.file_path);
            material = new THREE.MeshLambertMaterial({
                map: texture,
                transparent: true,
                side: THREE.DoubleSide
            });
        } else {
            // Use solid color
            const color = this.extractColorFromData(itemData.color);
            material = new THREE.MeshLambertMaterial({
                color: color,
                side: THREE.DoubleSide
            });
        }

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        return mesh;
    }

    getGeometryForClothingType(type) {
        const typeStr = type.toLowerCase();

        if (typeStr.includes('dress')) {
            return new THREE.PlaneGeometry(1.2, 1.8);
        } else if (typeStr.includes('trouser') || typeStr.includes('pant')) {
            return new THREE.PlaneGeometry(1.0, 1.2);
        } else if (typeStr.includes('coat') || typeStr.includes('jacket')) {
            return new THREE.PlaneGeometry(1.3, 1.1);
        } else if (typeStr.includes('shoe') || typeStr.includes('boot')) {
            return new THREE.PlaneGeometry(0.6, 0.8);
        } else {
            // Default for tops
            return new THREE.PlaneGeometry(1.2, 1.0);
        }
    }

    applyColorToMesh(mesh, colorData) {
        if (!colorData) return;

        let color = 0x808080; // Default gray

        if (typeof colorData === 'object' && colorData.rgb) {
            const [r, g, b] = colorData.rgb;
            color = new THREE.Color(r / 255, g / 255, b / 255);
        } else if (typeof colorData === 'string') {
            if (colorData.includes(' ')) {
                const colorValues = colorData.split(' ').map(v => parseInt(v.trim()));
                if (colorValues.length >= 3) {
                    const [r, g, b] = colorValues;
                    color = new THREE.Color(r / 255, g / 255, b / 255);
                }
            } else if (colorData.startsWith('#')) {
                color = new THREE.Color(colorData);
            }
        }

        mesh.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.color = color;
            }
        });
    }

    extractColorFromData(colorData) {
        if (!colorData) return 0x808080;

        if (typeof colorData === 'object' && colorData.rgb) {
            const [r, g, b] = colorData.rgb;
            return new THREE.Color(r / 255, g / 255, b / 255);
        } else if (typeof colorData === 'string') {
            if (colorData.includes(' ')) {
                const colorValues = colorData.split(' ').map(v => parseInt(v.trim()));
                if (colorValues.length >= 3) {
                    const [r, g, b] = colorValues;
                    return new THREE.Color(r / 255, g / 255, b / 255);
                }
            } else if (colorData.startsWith('#')) {
                return new THREE.Color(colorData);
            }
        }

        return new THREE.Color(0x808080);
    }

    positionClothingOnAvatar(clothingMesh, itemData) {
        if (!this.avatar) {
            console.warn('⚠️ No avatar loaded for positioning');
            return;
        }

        // Get avatar bounds
        const avatarBox = new THREE.Box3().setFromObject(this.avatar);
        const avatarHeight = avatarBox.max.y - avatarBox.min.y;
        const avatarCenter = avatarBox.getCenter(new THREE.Vector3());

        const clothingType = this.determineClothingType(itemData);

        // Position based on clothing type
        switch (clothingType) {
            case 'top':
                clothingMesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y + avatarHeight * 0.15,
                    avatarCenter.z + 0.1
                );
                break;
            case 'bottom':
                clothingMesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y - avatarHeight * 0.15,
                    avatarCenter.z + 0.1
                );
                break;
            case 'dress':
                clothingMesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y - avatarHeight * 0.05,
                    avatarCenter.z + 0.1
                );
                break;
            case 'outerwear':
                clothingMesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y + avatarHeight * 0.15,
                    avatarCenter.z + 0.15
                );
                break;
            case 'shoes':
                clothingMesh.position.set(
                    avatarCenter.x,
                    avatarBox.min.y + 0.05,
                    avatarCenter.z + 0.05
                );
                break;
            default:
                clothingMesh.position.copy(avatarCenter);
                clothingMesh.position.z += 0.1;
        }

        // Scale appropriately
        const scale = avatarHeight / 2.5;
        clothingMesh.scale.setScalar(scale);

        console.log(`📏 Positioned ${clothingType} at:`, clothingMesh.position);
    }

    determineClothingType(itemData) {
        const label = (itemData.label || itemData.type || '').toLowerCase();

        if (label.includes('shirt') || label.includes('top') || label.includes('pullover')) {
            return 'top';
        } else if (label.includes('trouser') || label.includes('pant')) {
            return 'bottom';
        } else if (label.includes('dress')) {
            return 'dress';
        } else if (label.includes('coat') || label.includes('jacket')) {
            return 'outerwear';
        } else if (label.includes('shoe') || label.includes('boot') || label.includes('sandal')) {
            return 'shoes';
        } else {
            return 'top'; // Default
        }
    }

    // Clothing manipulation methods
    resetClothingPosition(itemId) {
        const clothingData = this.currentClothing.get(itemId);
        if (!clothingData) return;

        clothingData.mesh.position.copy(clothingData.originalPosition);
        clothingData.mesh.rotation.copy(clothingData.originalRotation);
        clothingData.mesh.scale.copy(clothingData.originalScale);

        this.updatePositionInputs(clothingData.mesh);
        this.showMessage('Clothing position reset', 'info');
    }

    fitToAvatar(itemId) {
        const clothingData = this.currentClothing.get(itemId);
        if (!clothingData) return;

        // Re-position the clothing on the avatar
        this.positionClothingOnAvatar(clothingData.mesh, clothingData.itemData);
        this.updatePositionInputs(clothingData.mesh);
        this.showMessage('Clothing fitted to avatar', 'success');
    }

    saveClothingTransform(itemId) {
        const clothingData = this.currentClothing.get(itemId);
        if (!clothingData) return;

        // Update stored original transform
        clothingData.originalPosition.copy(clothingData.mesh.position);
        clothingData.originalRotation.copy(clothingData.mesh.rotation);
        clothingData.originalScale.copy(clothingData.mesh.scale);

        // Could save to database here if needed
        this.showMessage('Clothing transform saved', 'success');
    }

    clearAllClothing() {
        console.log('🗑️ Clearing all clothing...');

        for (const [itemId, clothingData] of this.currentClothing.entries()) {
            this.scene.remove(clothingData.mesh);

            // Dispose geometry and materials
            if (clothingData.mesh.geometry) {
                clothingData.mesh.geometry.dispose();
            }
            if (clothingData.mesh.material) {
                if (Array.isArray(clothingData.mesh.material)) {
                    clothingData.mesh.material.forEach(mat => mat.dispose());
                } else {
                    clothingData.mesh.material.dispose();
                }
            }
        }

        this.currentClothing.clear();
        this.deselectClothing();
        this.showMessage('All clothing cleared', 'info');
    }

    resetAllClothingPositions() {
        console.log('🔄 Resetting all clothing positions...');

        for (const [itemId, clothingData] of this.currentClothing.entries()) {
            clothingData.mesh.position.copy(clothingData.originalPosition);
            clothingData.mesh.rotation.copy(clothingData.originalRotation);
            clothingData.mesh.scale.copy(clothingData.originalScale);
        }

        this.showMessage('All clothing positions reset', 'info');
    }

    // Utility methods
    loadGLTF(path) {
        return new Promise((resolve, reject) => {
            this.gltfLoader.load(path, resolve, undefined, reject);
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const container = document.getElementById('avatar-container');
        if (!container) return;

        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }

    showMessage(message, type = 'info') {
        // Create and show a message toast
        const messageEl = document.createElement('div');
        messageEl.className = `message ${type}`;
        messageEl.innerHTML = `<i class="bi bi-${this.getMessageIcon(type)}"></i> ${message}`;

        document.body.appendChild(messageEl);

        setTimeout(() => {
            messageEl.remove();
        }, 3000);
    }

    getMessageIcon(type) {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-triangle',
            'info': 'info-circle',
            'warning': 'exclamation-circle'
        };
        return icons[type] || 'info-circle';
    }

    // Public API methods
    getLoadingStatus() {
        return {
            avatar: this.avatar !== null,
            clothingCount: this.currentClothing.size,
            selectedClothing: this.selectedClothing !== null
        };
    }

    getClothingList() {
        const items = [];
        for (const [itemId, clothingData] of this.currentClothing.entries()) {
            items.push({
                id: itemId,
                name: clothingData.itemData.label,
                type: this.determineClothingType(clothingData.itemData),
                position: clothingData.mesh.position.toArray(),
                rotation: clothingData.mesh.rotation.toArray(),
                scale: clothingData.mesh.scale.toArray()
            });
        }
        return items;
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for other systems to load
    setTimeout(() => {
        if (!window.enhancedAvatarSystem) {
            window.enhancedAvatarSystem = new EnhancedAvatarWardrobeSystem();
        }
    }, 1000);
});

// Also initialize if THREE.js loads later
window.addEventListener('load', function() {
    setTimeout(() => {
        if (!window.enhancedAvatarSystem && typeof THREE !== 'undefined') {
            window.enhancedAvatarSystem = new EnhancedAvatarWardrobeSystem();
        }
    }, 500);
});

console.log('📝 Enhanced Avatar Wardrobe System script loaded');