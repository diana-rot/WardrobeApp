// button-handlers.js - Event handlers for clothing-related buttons

document.addEventListener('DOMContentLoaded', function() {
    console.log('[Handlers] Initializing button handlers');

    // Handle Load Top.glb button click for custom avatar
    const loadTopGlbBtn = document.getElementById('load-top-glb');
    if (loadTopGlbBtn) {
        // Remove any existing event listeners
        const newBtn = loadTopGlbBtn.cloneNode(true);
        loadTopGlbBtn.parentNode.replaceChild(newBtn, loadTopGlbBtn);

        // Add our new event listener
        newBtn.addEventListener('click', async function() {
            console.log('[Handlers] Load Top.glb button clicked');
            try {
                const loadingElement = document.querySelector('.loading-overlay');
                if (loadingElement) loadingElement.style.display = 'flex';

                if (!window.avatar) {
                    console.log('[Handlers] No avatar found, loading female model first');
                    await loadAvatarModel({ gender: 'female' });
                }

                // Check if we should use the clothing manager
                if (window.avatarClothingManager) {
                    console.log('[Handlers] Using avatarClothingManager to load top.glb');
                    const testItem = {
                        id: 'top-glb',
                        type: 'tops',
                        label: 'T-shirt/top',
                        file_path: '/static/models/clothing/top.glb'
                    };

                    const result = await window.avatarClothingManager.applyClothing(testItem);

                    if (result) {
                        if (typeof showMessage === 'function')
                            showMessage('Top.glb loaded successfully', 'success');
                    } else {
                        if (typeof showMessage === 'function')
                            showMessage('Failed to load top.glb', 'error');
                    }
                } else {
                    // Fallback to original implementation
                    console.log('[Handlers] Falling back to original implementation');

                    const topModelPath = '/static/models/clothing/top.glb';
                    console.log('[Handlers] Loading top.glb from:', topModelPath);

                    const loader = new THREE.GLTFLoader();
                    const gltf = await new Promise((resolve, reject) => {
                        loader.load(
                            topModelPath,
                            resolve,
                            (xhr) => {
                                const progress = Math.floor((xhr.loaded / xhr.total) * 100);
                                console.log(`Loading top.glb: ${progress}%`);
                            },
                            reject
                        );
                    });

                    if (currentClothing['top'] && scene) {
                        scene.remove(currentClothing['top']);
                        currentClothing['top'] = null;
                    }

                    const topModel = gltf.scene;
                    topModel.position.set(0, 0.3, 0);
                    topModel.scale.set(0.8, 0.8, 0.8);

                    topModel.traverse((node) => {
                        if (node.isMesh) {
                            node.castShadow = true;
                            node.receiveShadow = true;
                        }
                    });

                    scene.add(topModel);
                    currentClothing['top'] = topModel;

                    if (typeof showMessage === 'function')
                        showMessage('Top.glb loaded successfully', 'success');
                }
            } catch (error) {
                console.error('[Handlers] Error loading top.glb:', error);
                if (typeof showMessage === 'function')
                    showMessage('Failed to load top.glb: ' + error.message, 'error');
            } finally {
                const loadingElement = document.querySelector('.loading-overlay');
                if (loadingElement) loadingElement.style.display = 'none';
            }
        });
    }

    // Replace test-custom-top button handler to use clothing manager
    const testCustomTopBtn = document.getElementById('test-custom-top');
    if (testCustomTopBtn) {
        // Remove any existing event listeners
        const newBtn = testCustomTopBtn.cloneNode(true);
        testCustomTopBtn.parentNode.replaceChild(newBtn, testCustomTopBtn);

        // Add our new event listener
        newBtn.addEventListener('click', async function() {
            console.log('[Handlers] Test Custom Top button clicked');
            try {
                const loadingElement = document.querySelector('.loading-overlay');
                if (loadingElement) loadingElement.style.display = 'flex';

                if (!window.avatar) {
                    console.log('[Handlers] No avatar found, loading female model first');
                    await loadAvatarModel({ gender: 'female' });
                }

                // Check if we should use the clothing manager
                if (window.avatarClothingManager) {
                    console.log('[Handlers] Using avatarClothingManager for test top');
                    const testItem = {
                        id: 'test-top',
                        type: 'tops',
                        label: 'T-shirt/top',
                        color: '#3498db'
                    };

                    const result = await window.avatarClothingManager.applyClothing(testItem);

                    if (result) {
                        if (typeof showMessage === 'function')
                            showMessage('Custom top applied successfully', 'success');
                    } else {
                        if (typeof showMessage === 'function')
                            showMessage('Failed to apply custom top', 'error');
                    }
                } else {
                    // Fallback to original implementation
                    console.log('[Handlers] Falling back to original implementation');
                    const testItem = {
                        id: 'test-top',
                        type: 'top',
                        color: '#3498db',
                        label: 'Test Top'
                    };

                    await applyClothing(testItem);
                    if (typeof showMessage === 'function')
                        showMessage('Custom top applied to avatar', 'success');
                }
            } catch (error) {
                console.error('[Handlers] Error testing custom top:', error);
                if (typeof showMessage === 'function')
                    showMessage('Failed to apply custom top: ' + error.message, 'error');
            } finally {
                const loadingElement = document.querySelector('.loading-overlay');
                if (loadingElement) loadingElement.style.display = 'none';
            }
        });
    }

    // Update the female model load button to properly initialize clothing manager
    const loadFemaleModelBtn = document.getElementById('load-female-model');
    if (loadFemaleModelBtn) {
        // Remove any existing event listeners
        const newBtn = loadFemaleModelBtn.cloneNode(true);
        loadFemaleModelBtn.parentNode.replaceChild(newBtn, loadFemaleModelBtn);

        // Add our new event listener
        newBtn.addEventListener('click', async function() {
            console.log('[Handlers] Load Female Model button clicked');
            try {
                // Show loading indicator
                const loadingElement = document.querySelector('.loading-overlay');
                if (loadingElement) {
                    loadingElement.style.display = 'flex';
                }

                // Initialize scene if needed
                if (!window.scene) {
                    console.log('[Handlers] Scene not initialized, calling init()');
                    if (typeof init === 'function') {
                        init();
                    } else {
                        throw new Error('Init function not available');
                    }
                }

                // Make sure avatarContainer is defined
                if (!window.avatarContainer) {
                    window.avatarContainer = document.getElementById('avatar-container');
                }

                if (!window.avatarContainer) {
                    console.error('[Handlers] Avatar container not found');
                    throw new Error('Avatar container not found');
                }

                // Load the female model
                console.log('[Handlers] Loading female avatar model');
                const success = await loadAvatarModel({ gender: 'female' });

                if (success) {
                    console.log('[Handlers] Female model loaded successfully');

                    // Update clothing manager references
                    if (window.avatarClothingManager) {
                        window.avatarClothingManager.setReferences(window.avatar, window.scene);
                        console.log('[Handlers] Updated clothing manager references');
                    }

                    // Set camera and controls
                    if (window.camera && window.controls) {
                        window.camera.position.set(0, 1.5, 4.0);
                        window.controls.target.set(0, 0.9, 0);
                        window.controls.update();
                    }

                    if (typeof showMessage === 'function')
                        showMessage('Female model loaded successfully', 'success');
                } else {
                    console.error('[Handlers] Failed to load female model');
                    if (typeof showMessage === 'function')
                        showMessage('Failed to load female model', 'error');
                }
            } catch (error) {
                console.error('[Handlers] Error loading female model:', error);
                if (typeof showMessage === 'function')
                    showMessage('Error: ' + error.message, 'error');
            } finally {
                // Hide loading indicator
                const loadingElement = document.querySelector('.loading-overlay');
                if (loadingElement) {
                    loadingElement.style.display = 'none';
                }
            }
        });
    }

    console.log('[Handlers] Button handlers initialized');
});