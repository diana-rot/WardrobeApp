// transform-controls-integration.js - Add THREE.js Transform Controls
// Add this script AFTER THREE.js is loaded

(function() {
    'use strict';

    console.log('🎮 Loading Transform Controls Integration...');

    // Load Transform Controls from CDN if not available
    function loadTransformControls() {
        return new Promise((resolve, reject) => {
            if (THREE.TransformControls) {
                console.log('✅ TransformControls already available');
                resolve(true);
                return;
            }

            console.log('📥 Loading TransformControls from CDN...');
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TransformControls.js';
            script.onload = () => {
                console.log('✅ TransformControls loaded successfully');
                resolve(true);
            };
            script.onerror = () => {
                console.error('❌ Failed to load TransformControls');
                reject(new Error('Failed to load TransformControls'));
            };
            document.head.appendChild(script);
        });
    }

    // Enhanced Clothing Item with Transform Controls
    class ClothingItemWithControls {
        constructor(mesh, itemData, scene, camera, renderer) {
            this.mesh = mesh;
            this.itemData = itemData;
            this.scene = scene;
            this.camera = camera;
            this.renderer = renderer;

            this.transformControls = null;
            this.isSelected = false;
            this.originalTransform = {
                position: mesh.position.clone(),
                rotation: mesh.rotation.clone(),
                scale: mesh.scale.clone()
            };

            this.setupTransformControls();
        }

        async setupTransformControls() {
            try {
                // Ensure TransformControls is loaded
                await loadTransformControls();

                // Create transform controls
                this.transformControls = new THREE.TransformControls(this.camera, this.renderer.domElement);
                this.transformControls.setMode('translate'); // Start with translate mode
                this.transformControls.setSize(0.8); // Smaller handles

                // Add to scene
                this.scene.add(this.transformControls);

                // Setup event listeners
                this.setupControlsEvents();

                console.log('✅ Transform controls setup complete for:', this.itemData.label);

            } catch (error) {
                console.error('❌ Error setting up transform controls:', error);
            }
        }

        setupControlsEvents() {
            if (!this.transformControls) return;

            // Handle transform changes
            this.transformControls.addEventListener('change', () => {
                this.renderer.render(this.scene, this.camera);
            });

            // Handle dragging state
            this.transformControls.addEventListener('dragging-changed', (event) => {
                // Disable orbit controls while dragging
                if (window.avatarManager && window.avatarManager.controls) {
                    window.avatarManager.controls.enabled = !event.value;
                }
                if (window.enhancedAvatarSystem && window.enhancedAvatarSystem.controls) {
                    window.enhancedAvatarSystem.controls.enabled = !event.value;
                }
            });

            // Handle selection
            this.transformControls.addEventListener('objectChange', () => {
                this.updateControlsUI();
            });
        }

        select() {
            if (!this.transformControls) return;

            this.isSelected = true;
            this.transformControls.attach(this.mesh);
            this.highlightMesh(true);
            this.showControlsPanel();

            console.log('👕 Selected clothing item:', this.itemData.label);
        }

        deselect() {
            if (!this.transformControls) return;

            this.isSelected = false;
            this.transformControls.detach();
            this.highlightMesh(false);
            this.hideControlsPanel();
        }

        highlightMesh(highlight) {
            if (highlight) {
                // Add highlight effect
                this.mesh.traverse((child) => {
                    if (child.isMesh && child.material) {
                        if (!child.userData.originalMaterial) {
                            child.userData.originalMaterial = child.material.clone();
                        }
                        child.material = child.material.clone();
                        child.material.emissive = new THREE.Color(0x444444);
                    }
                });
            } else {
                // Remove highlight effect
                this.mesh.traverse((child) => {
                    if (child.isMesh && child.userData.originalMaterial) {
                        child.material = child.userData.originalMaterial;
                        delete child.userData.originalMaterial;
                    }
                });
            }
        }

        showControlsPanel() {
            this.hideControlsPanel(); // Remove any existing panel

            const panel = document.createElement('div');
            panel.id = 'clothing-transform-panel';
            panel.className = 'clothing-transform-panel';
            panel.innerHTML = this.getControlsPanelHTML();

            document.body.appendChild(panel);
            this.setupPanelEvents();
            this.updateControlsUI();
        }

        hideControlsPanel() {
            const existingPanel = document.getElementById('clothing-transform-panel');
            if (existingPanel) {
                existingPanel.remove();
            }
        }

        getControlsPanelHTML() {
            return `
                <div class="transform-panel-header">
                    <h4>🎮 ${this.itemData.label || 'Clothing Item'}</h4>
                    <button class="close-panel" onclick="clothingTransformManager.deselectAll()">×</button>
                </div>
                <div class="transform-panel-body">
                    <div class="control-section">
                        <label>Transform Mode:</label>
                        <div class="mode-buttons">
                            <button class="mode-btn active" data-mode="translate">📐 Move</button>
                            <button class="mode-btn" data-mode="rotate">🔄 Rotate</button>
                            <button class="mode-btn" data-mode="scale">📏 Scale</button>
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <label>Position:</label>
                        <div class="input-group">
                            <input type="number" id="pos-x" step="0.01" placeholder="X">
                            <input type="number" id="pos-y" step="0.01" placeholder="Y">
                            <input type="number" id="pos-z" step="0.01" placeholder="Z">
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <label>Rotation (degrees):</label>
                        <div class="input-group">
                            <input type="number" id="rot-x" step="1" placeholder="X">
                            <input type="number" id="rot-y" step="1" placeholder="Y">
                            <input type="number" id="rot-z" step="1" placeholder="Z">
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <label>Scale:</label>
                        <div class="input-group">
                            <input type="number" id="scale-x" step="0.1" placeholder="X" min="0.1">
                            <input type="number" id="scale-y" step="0.1" placeholder="Y" min="0.1">
                            <input type="number" id="scale-z" step="0.1" placeholder="Z" min="0.1">
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <div class="action-buttons">
                            <button class="action-btn reset-btn" onclick="clothingTransformManager.resetSelected()">↺ Reset</button>
                            <button class="action-btn fit-btn" onclick="clothingTransformManager.fitToAvatar()">👤 Fit to Avatar</button>
                            <button class="action-btn save-btn" onclick="clothingTransformManager.saveTransform()">💾 Save</button>
                        </div>
                    </div>
                </div>
            `;
        }

        setupPanelEvents() {
            const panel = document.getElementById('clothing-transform-panel');
            if (!panel) return;

            // Mode buttons
            panel.querySelectorAll('.mode-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    panel.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');

                    const mode = btn.dataset.mode;
                    this.transformControls.setMode(mode);

                    console.log('🎮 Transform mode changed to:', mode);
                });
            });

            // Position inputs
            ['x', 'y', 'z'].forEach((axis, index) => {
                const input = panel.querySelector(`#pos-${axis}`);
                if (input) {
                    input.addEventListener('input', () => {
                        const value = parseFloat(input.value) || 0;
                        this.mesh.position.setComponent(index, value);
                        this.renderer.render(this.scene, this.camera);
                    });
                }
            });

            // Rotation inputs
            ['x', 'y', 'z'].forEach((axis, index) => {
                const input = panel.querySelector(`#rot-${axis}`);
                if (input) {
                    input.addEventListener('input', () => {
                        const value = (parseFloat(input.value) || 0) * Math.PI / 180; // Convert to radians
                        this.mesh.rotation.setComponent(index, value);
                        this.renderer.render(this.scene, this.camera);
                    });
                }
            });

            // Scale inputs
            ['x', 'y', 'z'].forEach((axis, index) => {
                const input = panel.querySelector(`#scale-${axis}`);
                if (input) {
                    input.addEventListener('input', () => {
                        const value = Math.max(0.1, parseFloat(input.value) || 1);
                        this.mesh.scale.setComponent(index, value);
                        this.renderer.render(this.scene, this.camera);
                    });
                }
            });
        }

        updateControlsUI() {
            const panel = document.getElementById('clothing-transform-panel');
            if (!panel) return;

            // Update position inputs
            ['x', 'y', 'z'].forEach((axis, index) => {
                const input = panel.querySelector(`#pos-${axis}`);
                if (input) {
                    input.value = this.mesh.position.getComponent(index).toFixed(2);
                }
            });

            // Update rotation inputs (convert to degrees)
            ['x', 'y', 'z'].forEach((axis, index) => {
                const input = panel.querySelector(`#rot-${axis}`);
                if (input) {
                    input.value = Math.round(this.mesh.rotation.getComponent(index) * 180 / Math.PI);
                }
            });

            // Update scale inputs
            ['x', 'y', 'z'].forEach((axis, index) => {
                const input = panel.querySelector(`#scale-${axis}`);
                if (input) {
                    input.value = this.mesh.scale.getComponent(index).toFixed(2);
                }
            });
        }

        resetTransform() {
            this.mesh.position.copy(this.originalTransform.position);
            this.mesh.rotation.copy(this.originalTransform.rotation);
            this.mesh.scale.copy(this.originalTransform.scale);

            this.updateControlsUI();
            this.renderer.render(this.scene, this.camera);

            console.log('↺ Transform reset for:', this.itemData.label);
        }

        fitToAvatar() {
            // Re-position the clothing item on the avatar
            if (window.clothingOBJRenderer) {
                const clothingType = window.clothingOBJRenderer.determineClothingType(this.itemData);
                window.clothingOBJRenderer.positionClothingOnAvatar(this.mesh, clothingType);
            } else if (window.enhancedAvatarSystem) {
                window.enhancedAvatarSystem.positionClothingOnAvatar(this.mesh, this.itemData);
            }

            this.updateControlsUI();
            this.renderer.render(this.scene, this.camera);

            console.log('👤 Fitted to avatar:', this.itemData.label);
        }

        saveTransform() {
            // Update original transform
            this.originalTransform.position.copy(this.mesh.position);
            this.originalTransform.rotation.copy(this.mesh.rotation);
            this.originalTransform.scale.copy(this.mesh.scale);

            console.log('💾 Transform saved for:', this.itemData.label);

            // Show confirmation
            this.showSaveConfirmation();
        }

        showSaveConfirmation() {
            const confirmation = document.createElement('div');
            confirmation.className = 'save-confirmation';
            confirmation.textContent = '✅ Transform saved!';
            confirmation.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: #28a745;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: 500;
                z-index: 10000;
                animation: fadeInOut 2s ease-in-out;
            `;

            document.body.appendChild(confirmation);
            setTimeout(() => {
                confirmation.remove();
            }, 2000);
        }

        dispose() {
            if (this.transformControls) {
                this.scene.remove(this.transformControls);
                this.transformControls.dispose();
            }
            this.hideControlsPanel();
        }
    }

    // Global Clothing Transform Manager
    class ClothingTransformManager {
        constructor() {
            this.clothingItems = new Map();
            this.selectedItem = null;
            this.scene = null;
            this.camera = null;
            this.renderer = null;
            this.raycaster = new THREE.Raycaster();
            this.mouse = new THREE.Vector2();

            this.setupStyles();
            this.setupEventListeners();
        }

        initialize(scene, camera, renderer) {
            this.scene = scene;
            this.camera = camera;
            this.renderer = renderer;

            console.log('✅ ClothingTransformManager initialized');
        }

        addClothingItem(itemId, mesh, itemData) {
            if (!this.scene || !this.camera || !this.renderer) {
                console.warn('⚠️ Transform manager not initialized');
                return;
            }

            const clothingItem = new ClothingItemWithControls(
                mesh, itemData, this.scene, this.camera, this.renderer
            );

            this.clothingItems.set(itemId, clothingItem);
            console.log('➕ Added clothing item to transform manager:', itemData.label);
        }

        removeClothingItem(itemId) {
            const clothingItem = this.clothingItems.get(itemId);
            if (clothingItem) {
                if (this.selectedItem === clothingItem) {
                    this.deselectAll();
                }
                clothingItem.dispose();
                this.clothingItems.delete(itemId);
                console.log('➖ Removed clothing item from transform manager');
            }
        }

        selectClothingItem(itemId) {
            this.deselectAll();

            const clothingItem = this.clothingItems.get(itemId);
            if (clothingItem) {
                this.selectedItem = clothingItem;
                clothingItem.select();
            }
        }

        deselectAll() {
            if (this.selectedItem) {
                this.selectedItem.deselect();
                this.selectedItem = null;
            }
        }

        resetSelected() {
            if (this.selectedItem) {
                this.selectedItem.resetTransform();
            }
        }

        fitToAvatar() {
            if (this.selectedItem) {
                this.selectedItem.fitToAvatar();
            }
        }

        saveTransform() {
            if (this.selectedItem) {
                this.selectedItem.saveTransform();
            }
        }

        setupEventListeners() {
            document.addEventListener('click', (event) => {
                // Only handle clicks in avatar container
                const avatarContainer = document.getElementById('avatar-container');
                if (!avatarContainer || !avatarContainer.contains(event.target)) {
                    return;
                }

                // Check if we're in control mode
                const controlModeBtn = document.querySelector('.avatar-mode-btn[data-mode="try-on"]');
                if (!controlModeBtn || !controlModeBtn.classList.contains('active')) {
                    return;
                }

                this.handleClick(event);
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', (event) => {
                if (!this.selectedItem) return;

                switch (event.key) {
                    case 'g':
                        this.selectedItem.transformControls.setMode('translate');
                        break;
                    case 'r':
                        this.selectedItem.transformControls.setMode('rotate');
                        break;
                    case 's':
                        this.selectedItem.transformControls.setMode('scale');
                        break;
                    case 'Escape':
                        this.deselectAll();
                        break;
                }
            });
        }

        handleClick(event) {
            if (!this.renderer || !this.camera) return;

            const rect = this.renderer.domElement.getBoundingClientRect();
            this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            this.raycaster.setFromCamera(this.mouse, this.camera);

            // Check for clothing intersections
            const clothingMeshes = Array.from(this.clothingItems.values()).map(item => item.mesh);
            const intersects = this.raycaster.intersectObjects(clothingMeshes, true);

            if (intersects.length > 0) {
                const selectedMesh = intersects[0].object;

                // Find which clothing item this mesh belongs to
                for (const [itemId, clothingItem] of this.clothingItems.entries()) {
                    if (clothingItem.mesh === selectedMesh ||
                        clothingItem.mesh.children.includes(selectedMesh) ||
                        this.isChildOfMesh(selectedMesh, clothingItem.mesh)) {
                        this.selectClothingItem(itemId);
                        break;
                    }
                }
            } else {
                this.deselectAll();
            }
        }

        isChildOfMesh(child, parent) {
            let current = child.parent;
            while (current) {
                if (current === parent) return true;
                current = current.parent;
            }
            return false;
        }

        setupStyles() {
            if (document.getElementById('transform-controls-styles')) return;

            const style = document.createElement('style');
            style.id = 'transform-controls-styles';
            style.textContent = `
                .clothing-transform-panel {
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    width: 280px;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                    backdrop-filter: blur(10px);
                    z-index: 1000;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }

                .transform-panel-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 16px;
                    border-bottom: 1px solid rgba(0,0,0,0.1);
                }

                .transform-panel-header h4 {
                    margin: 0;
                    font-size: 16px;
                    font-weight: 600;
                    color: #333;
                }

                .close-panel {
                    background: none;
                    border: none;
                    font-size: 20px;
                    color: #666;
                    cursor: pointer;
                    padding: 4px;
                    border-radius: 4px;
                }

                .close-panel:hover {
                    background: rgba(0,0,0,0.1);
                }

                .transform-panel-body {
                    padding: 16px;
                }

                .control-section {
                    margin-bottom: 16px;
                }

                .control-section label {
                    display: block;
                    font-size: 12px;
                    font-weight: 600;
                    color: #666;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                .mode-buttons {
                    display: flex;
                    gap: 4px;
                }

                .mode-btn {
                    flex: 1;
                    background: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    padding: 8px 6px;
                    font-size: 11px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }

                .mode-btn:hover {
                    background: #e9e9e9;
                    border-color: #bbb;
                }

                .mode-btn.active {
                    background: #3498db;
                    color: white;
                    border-color: #3498db;
                }

                .input-group {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 6px;
                }

                .input-group input {
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 12px;
                    text-align: center;
                }

                .input-group input:focus {
                    outline: none;
                    border-color: #3498db;
                    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
                }

                .action-buttons {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 6px;
                }

                .action-btn {
                    padding: 8px 6px;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    border: 1px solid;
                }

                .reset-btn {
                    background: #6c757d;
                    color: white;
                    border-color: #6c757d;
                }

                .reset-btn:hover {
                    background: #5a6268;
                }

                .fit-btn {
                    background: #17a2b8;
                    color: white;
                    border-color: #17a2b8;
                }

                .fit-btn:hover {
                    background: #138496;
                }

                .save-btn {
                    background: #28a745;
                    color: white;
                    border-color: #28a745;
                }

                .save-btn:hover {
                    background: #218838;
                }

                @keyframes fadeInOut {
                    0% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
                    20% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                    80% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                    100% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
                }
            `;
            document.head.appendChild(style);
        }
    }

    // Create global instance
    window.clothingTransformManager = new ClothingTransformManager();

    // Integration with existing systems
    function integrateWithExistingSystems() {
        // Integration with clothing renderer
        if (window.clothingOBJRenderer) {
            const originalLoadClothing = window.clothingOBJRenderer.loadClothingFromDatabase;
            window.clothingOBJRenderer.loadClothingFromDatabase = async function(itemId) {
                const result = await originalLoadClothing.call(this, itemId);

                if (result && this.currentClothing.has(itemId)) {
                    const clothingData = this.currentClothing.get(itemId);
                    window.clothingTransformManager.addClothingItem(
                        itemId, clothingData.mesh, clothingData.itemData
                    );
                }

                return result;
            };

            const originalRemoveClothing = window.clothingOBJRenderer.removeClothing;
            window.clothingOBJRenderer.removeClothing = async function(itemId) {
                window.clothingTransformManager.removeClothingItem(itemId);
                return await originalRemoveClothing.call(this, itemId);
            };
        }

        // Integration with enhanced avatar system
        if (window.enhancedAvatarSystem) {
            const originalLoadClothingItem = window.enhancedAvatarSystem.loadClothingItem;
            if (originalLoadClothingItem) {
                window.enhancedAvatarSystem.loadClothingItem = async function(itemId) {
                    const result = await originalLoadClothingItem.call(this, itemId);

                    if (result && this.currentClothing && this.currentClothing.has(itemId)) {
                        const clothingData = this.currentClothing.get(itemId);
                        window.clothingTransformManager.addClothingItem(
                            itemId, clothingData.mesh, clothingData.itemData
                        );
                    }

                    return result;
                };
            }
        }

        // Initialize transform manager when avatar systems are ready
        const initTransformManager = () => {
            let scene, camera, renderer;

            if (window.avatarManager) {
                scene = window.avatarManager.scene;
                camera = window.avatarManager.camera;
                renderer = window.avatarManager.renderer;
            } else if (window.enhancedAvatarSystem) {
                scene = window.enhancedAvatarSystem.scene;
                camera = window.enhancedAvatarSystem.camera;
                renderer = window.enhancedAvatarSystem.renderer;
            }

            if (scene && camera && renderer) {
                window.clothingTransformManager.initialize(scene, camera, renderer);
                console.log('✅ Transform manager integrated with avatar system');
            }
        };

        // Try to initialize immediately
        initTransformManager();

        // Also try after delays
        setTimeout(initTransformManager, 1000);
        setTimeout(initTransformManager, 3000);
        setTimeout(initTransformManager, 5000);
    }

    // Initialize integration
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', integrateWithExistingSystems);
    } else {
        integrateWithExistingSystems();
    }

    // Also try after a delay
    setTimeout(integrateWithExistingSystems, 2000);

    console.log('✅ Transform Controls Integration loaded');

})();