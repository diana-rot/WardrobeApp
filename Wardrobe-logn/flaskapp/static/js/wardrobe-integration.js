// wardrobe-integration-fixed.js - Fixed version for clothing loading
// Replace your existing wardrobe-integration.js with this version

(function() {
    'use strict';

    console.log('🚀 Initializing FIXED Wardrobe Avatar Integration...');

    // Global state
    let avatarSystem = null;
    let clothingRenderer = null;
    let isAvatarMode = false;
    let selectedClothingItems = new Set();
    let currentCategory = 'tops';

    // Wait for all systems to be ready
    function waitForSystems() {
        return new Promise((resolve) => {
            const checkSystems = () => {
                const hasAvatarManager = window.avatarManager && window.avatarManager.scene && window.avatarManager.avatarModel;
                const hasClothingRenderer = window.clothingOBJRenderer;
                const hasEnhancedSystem = window.enhancedAvatarSystem;

                console.log('📊 System status:', {
                    hasAvatarManager,
                    hasClothingRenderer,
                    hasEnhancedSystem
                });

                if (hasAvatarManager && hasClothingRenderer) {
                    avatarSystem = window.avatarManager;
                    clothingRenderer = window.clothingOBJRenderer;
                    console.log('✅ Using Avatar Manager + Clothing Renderer');
                    resolve(true);
                } else if (hasEnhancedSystem) {
                    avatarSystem = window.enhancedAvatarSystem;
                    console.log('✅ Using Enhanced Avatar System');
                    resolve(true);
                } else {
                    console.log('⏳ Waiting for avatar systems...');
                    setTimeout(checkSystems, 1000);
                }
            };

            checkSystems();
        });
    }

    // Initialize wardrobe integration
    async function initializeWardrobeIntegration() {
        console.log('🔧 Setting up wardrobe integration...');

        try {
            // Wait for systems to be ready
            await waitForSystems();

            // Setup wardrobe UI enhancements
            setupWardrobeUI();

            // Setup clothing item interactions
            setupClothingItemInteractions();

            // Setup avatar controls
            setupAvatarControls();

            // Load initial clothing items
            await loadClothingItems(currentCategory);

            // Setup category change listener
            setupCategoryListener();

            console.log('✅ Wardrobe integration initialized successfully');

        } catch (error) {
            console.error('❌ Error initializing wardrobe integration:', error);
        }
    }

    // FIXED: Load clothing items from API
    async function loadClothingItems(category) {
        console.log(`👕 Loading clothing items for category: ${category}`);

        const container = document.getElementById('wardrobe-items');
        if (!container) {
            console.error('❌ Wardrobe items container not found');
            return;
        }

        try {
            // Show loading state
            container.innerHTML = '<div class="loading">Loading wardrobe items...</div>';

            // Fetch items from API
            const response = await fetch(`/api/wardrobe/items?category=${category}`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('📦 Wardrobe API response:', data);

            if (data.success && data.items && data.items.length > 0) {
                let html = '';

                data.items.forEach(item => {
                    const itemId = item._id || item.id;
                    const itemName = item.label || item.name || 'Unknown Item';
                    const imageUrl = item.file_path || item.image_url || '/static/images/placeholder.png';

                    html += `
                        <div class="wardrobe-item" data-item-id="${itemId}" data-category="${category}">
                            <div class="item-image">
                                <img src="${imageUrl}" alt="${itemName}" onerror="this.src='/static/images/placeholder.png'">
                            </div>
                            <div class="item-name">${itemName}</div>
                        </div>
                    `;
                });

                container.innerHTML = html;
                console.log(`✅ Loaded ${data.items.length} items for category: ${category}`);

            } else {
                container.innerHTML = '<div class="no-items">No items found in this category</div>';
                console.log(`ℹ️ No items found for category: ${category}`);
            }

        } catch (error) {
            console.error('❌ Error loading clothing items:', error);
            container.innerHTML = `<div class="error">Failed to load items: ${error.message}</div>`;
        }
    }

    // Setup category change listener
    function setupCategoryListener() {
        const categorySelect = document.getElementById('category-wardrobe');
        if (categorySelect) {
            categorySelect.addEventListener('change', async function() {
                currentCategory = this.value;
                console.log(`🔄 Category changed to: ${currentCategory}`);
                await loadClothingItems(currentCategory);
            });
        }
    }

    function setupWardrobeUI() {
        console.log('🎨 Setting up wardrobe UI...');

        // Add avatar mode toggle to wardrobe tab
        const wardrobeSection = document.querySelector('#wardrobe .wardrobe-section');
        if (!wardrobeSection) {
            console.warn('⚠️ Wardrobe section not found');
            return;
        }

        // Create avatar controls panel
        const avatarControlsHTML = `
            <div class="avatar-wardrobe-controls" style="margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
                <h5 style="margin: 0 0 12px 0; color: #495057; font-size: 14px; font-weight: 600;">
                    👤 Avatar Try-On System
                </h5>
                <div class="avatar-mode-toggle" style="display: flex; background: #e9ecef; border-radius: 25px; padding: 3px; margin-bottom: 10px;">
                    <button class="avatar-mode-btn active" data-mode="browse" style="flex: 1; padding: 8px 12px; border: none; background: none; border-radius: 22px; cursor: pointer; font-weight: 500; transition: all 0.3s ease; color: #6c757d; font-size: 12px;">
                        📋 Browse Mode
                    </button>
                    <button class="avatar-mode-btn" data-mode="try-on" style="flex: 1; padding: 8px 12px; border: none; background: none; border-radius: 22px; cursor: pointer; font-weight: 500; transition: all 0.3s ease; color: #6c757d; font-size: 12px;">
                        👕 Try-On Mode
                    </button>
                </div>
                <div id="avatar-mode-info" class="mode-info" style="font-size: 11px; color: #6c757d; text-align: center;">
                    Browse your wardrobe items normally
                </div>
                <div id="try-on-controls" class="try-on-controls" style="display: none; margin-top: 10px;">
                    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                        <button id="clear-avatar-clothing" class="btn" style="flex: 1; padding: 6px 8px; background: #6c757d; color: white; border: none; border-radius: 4px; font-size: 11px; cursor: pointer;">
                            🗑️ Clear All
                        </button>
                        <button id="save-avatar-outfit" class="btn" style="flex: 1; padding: 6px 8px; background: #28a745; color: white; border: none; border-radius: 4px; font-size: 11px; cursor: pointer;">
                            💾 Save Outfit
                        </button>
                    </div>
                    <div id="selected-items-count" style="font-size: 11px; color: #495057; text-align: center;">
                        0 items selected
                    </div>
                </div>
            </div>
        `;

        // Insert avatar controls after the category selection
        const categorySelect = wardrobeSection.querySelector('#category-wardrobe')?.parentElement;
        if (categorySelect) {
            categorySelect.insertAdjacentHTML('afterend', avatarControlsHTML);
        } else {
            wardrobeSection.insertAdjacentHTML('afterbegin', avatarControlsHTML);
        }

        // Add custom styles
        addWardrobeStyles();

        // Setup mode toggle event listeners
        setupModeToggle();
    }

    function addWardrobeStyles() {
        if (document.getElementById('wardrobe-avatar-styles')) return;

        const style = document.createElement('style');
        style.id = 'wardrobe-avatar-styles';
        style.textContent = `
            .avatar-mode-btn.active {
                background: #3498db !important;
                color: white !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .avatar-mode-btn:hover {
                background: #e9ecef !important;
                color: #495057 !important;
            }

            .avatar-mode-btn.active:hover {
                background: #2980b9 !important;
                color: white !important;
            }

            .wardrobe-item.try-on-mode {
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .wardrobe-item.try-on-mode:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 25px rgba(52, 152, 219, 0.3) !important;
                border-color: #3498db !important;
            }

            .wardrobe-item.selected-for-avatar {
                border-color: #28a745 !important;
                background: #d4edda !important;
                box-shadow: 0 0 0 2px rgba(40, 167, 69, 0.3);
            }

            .wardrobe-item.selected-for-avatar::after {
                content: '✓';
                position: absolute;
                top: 8px;
                right: 8px;
                background: #28a745;
                color: white;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
                z-index: 2;
            }

            .avatar-loading-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(255, 255, 255, 0.9);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10;
                border-radius: 8px;
            }

            .avatar-loading-spinner {
                width: 20px;
                height: 20px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }

    function setupModeToggle() {
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('avatar-mode-btn')) {
                const mode = e.target.dataset.mode;
                switchAvatarMode(mode);
            }
        });

        // Setup control buttons
        document.addEventListener('click', function(e) {
            if (e.target.id === 'clear-avatar-clothing') {
                clearAvatarClothing();
            } else if (e.target.id === 'save-avatar-outfit') {
                saveAvatarOutfit();
            }
        });
    }

    function switchAvatarMode(mode) {
        console.log(`🔄 Switching to ${mode} mode`);

        // Update button states
        document.querySelectorAll('.avatar-mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        const modeInfo = document.getElementById('avatar-mode-info');
        const tryOnControls = document.getElementById('try-on-controls');

        if (mode === 'try-on') {
            isAvatarMode = true;
            modeInfo.textContent = 'Click clothing items to try them on your avatar';
            tryOnControls.style.display = 'block';

            // Add try-on class to wardrobe items
            document.querySelectorAll('.wardrobe-item').forEach(item => {
                item.classList.add('try-on-mode');
            });

            showMessage('Avatar try-on mode activated - click items to wear them!', 'info');
        } else {
            isAvatarMode = false;
            modeInfo.textContent = 'Browse your wardrobe items normally';
            tryOnControls.style.display = 'none';

            // Remove try-on class from wardrobe items
            document.querySelectorAll('.wardrobe-item').forEach(item => {
                item.classList.remove('try-on-mode', 'selected-for-avatar');
            });

            selectedClothingItems.clear();
            updateSelectedItemsCount();
            showMessage('Browse mode activated', 'info');
        }
    }

    function setupClothingItemInteractions() {
        console.log('👕 Setting up clothing item interactions...');

        // Add click handlers to wardrobe items
        document.addEventListener('click', async function(e) {
            const wardrobeItem = e.target.closest('.wardrobe-item');
            if (!wardrobeItem || !isAvatarMode) return;

            e.preventDefault();
            e.stopPropagation();

            const itemId = wardrobeItem.dataset.itemId;
            if (!itemId) {
                console.warn('⚠️ No item ID found for wardrobe item');
                return;
            }

            console.log(`👕 Clicked clothing item: ${itemId}`);

            if (wardrobeItem.classList.contains('selected-for-avatar')) {
                // Remove from avatar
                await removeClothingFromAvatar(itemId, wardrobeItem);
            } else {
                // Add to avatar
                await addClothingToAvatar(itemId, wardrobeItem);
            }
        });
    }

    async function addClothingToAvatar(itemId, element) {
        console.log(`➕ Adding clothing ${itemId} to avatar`);

        // Show loading state
        showItemLoading(element, true);

        try {
            let success = false;

            if (clothingRenderer && avatarSystem) {
                // FIXED: Ensure references are set properly
                const referencesSet = await clothingRenderer.setReferences(avatarSystem.scene, avatarSystem.avatarModel);
                if (referencesSet) {
                    success = await clothingRenderer.loadClothingFromDatabase(itemId);
                } else {
                    throw new Error('Failed to set clothing renderer references');
                }
            } else if (avatarSystem && avatarSystem.loadClothingItem) {
                // Use enhanced avatar system
                success = await avatarSystem.loadClothingItem(itemId);
            } else {
                throw new Error('No avatar system available');
            }

            if (success) {
                element.classList.add('selected-for-avatar');
                selectedClothingItems.add(itemId);
                updateSelectedItemsCount();
                showMessage('Clothing added to avatar!', 'success');
            } else {
                throw new Error('Failed to load clothing on avatar');
            }

        } catch (error) {
            console.error(`❌ Error adding clothing ${itemId}:`, error);
            showMessage(`Failed to add clothing: ${error.message}`, 'error');
        } finally {
            showItemLoading(element, false);
        }
    }

    async function removeClothingFromAvatar(itemId, element) {
        console.log(`➖ Removing clothing ${itemId} from avatar`);

        try {
            let success = false;

            if (clothingRenderer) {
                success = await clothingRenderer.removeClothing(itemId);
            } else if (avatarSystem && avatarSystem.removeClothingItem) {
                success = await avatarSystem.removeClothingItem(itemId);
            } else if (avatarSystem && avatarSystem.currentClothing) {
                // Manual removal for enhanced system
                const clothingData = avatarSystem.currentClothing.get(itemId);
                if (clothingData) {
                    avatarSystem.scene.remove(clothingData.mesh);
                    avatarSystem.currentClothing.delete(itemId);
                    success = true;
                }
            }

            if (success) {
                element.classList.remove('selected-for-avatar');
                selectedClothingItems.delete(itemId);
                updateSelectedItemsCount();
                showMessage('Clothing removed from avatar', 'info');
            }

        } catch (error) {
            console.error(`❌ Error removing clothing ${itemId}:`, error);
            showMessage(`Failed to remove clothing: ${error.message}`, 'error');
        }
    }

    function showItemLoading(element, show) {
        if (show) {
            const overlay = document.createElement('div');
            overlay.className = 'avatar-loading-overlay';
            overlay.innerHTML = '<div class="avatar-loading-spinner"></div>';
            element.style.position = 'relative';
            element.appendChild(overlay);
        } else {
            const overlay = element.querySelector('.avatar-loading-overlay');
            if (overlay) {
                overlay.remove();
            }
        }
    }

    function updateSelectedItemsCount() {
        const countElement = document.getElementById('selected-items-count');
        if (countElement) {
            const count = selectedClothingItems.size;
            countElement.textContent = `${count} item${count !== 1 ? 's' : ''} selected`;
        }
    }

    async function clearAvatarClothing() {
        console.log('🗑️ Clearing all avatar clothing...');

        try {
            if (clothingRenderer) {
                await clothingRenderer.clearAllClothing();
            } else if (avatarSystem && avatarSystem.clearAllClothing) {
                await avatarSystem.clearAllClothing();
            } else if (avatarSystem && avatarSystem.currentClothing) {
                // Manual clearing for enhanced system
                for (const [itemId, clothingData] of avatarSystem.currentClothing.entries()) {
                    avatarSystem.scene.remove(clothingData.mesh);
                }
                avatarSystem.currentClothing.clear();
            }

            // Reset UI
            document.querySelectorAll('.wardrobe-item').forEach(item => {
                item.classList.remove('selected-for-avatar');
            });

            selectedClothingItems.clear();
            updateSelectedItemsCount();
            showMessage('All clothing removed from avatar', 'info');

        } catch (error) {
            console.error('❌ Error clearing avatar clothing:', error);
            showMessage('Failed to clear clothing', 'error');
        }
    }

    // FIXED: Save Avatar Outfit functionality
    async function saveAvatarOutfit() {
        console.log('💾 Saving avatar outfit...');

        if (selectedClothingItems.size === 0) {
            showMessage('No clothing items selected to save', 'warning');
            return;
        }

        try {
            // Prepare outfit data
            const outfitData = {
                name: `Outfit ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`,
                items: Array.from(selectedClothingItems),
                created_at: new Date().toISOString(),
                type: 'avatar_outfit',
                avatar_config: getAvatarConfiguration()
            };

            // Send to server
            const response = await fetch('/api/outfits/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(outfitData)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                showMessage(`Outfit "${outfitData.name}" saved successfully!`, 'success');
                console.log('💾 Outfit saved:', result);
            } else {
                throw new Error(result.error || 'Failed to save outfit');
            }

        } catch (error) {
            console.error('❌ Error saving outfit:', error);
            showMessage('Failed to save outfit: ' + error.message, 'error');
        }
    }

    // Get current avatar configuration
    function getAvatarConfiguration() {
        if (avatarSystem && avatarSystem.getConfiguration) {
            return avatarSystem.getConfiguration();
        } else if (window.avatarCustomization) {
            return window.avatarCustomization;
        } else {
            return {
                gender: 'female',
                bodySize: 'm',
                height: 'medium',
                skinColor: 'light',
                eyeColor: 'brown',
                hairType: 'elvis_hazel',
                hairColor: 'brown'
            };
        }
    }

    function setupAvatarControls() {
        console.log('🎮 Setting up avatar controls...');

        // Listen for clothing renderer ready event
        window.addEventListener('clothingRendererReady', (event) => {
            console.log('👕 Clothing renderer is ready!', event.detail);
            clothingRenderer = event.detail.renderer;
        });

        // Setup wardrobe item ID detection for dynamically loaded items
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        const wardrobeItems = node.querySelectorAll ? node.querySelectorAll('.wardrobe-item') : [];
                        wardrobeItems.forEach(setupItemId);

                        if (node.classList && node.classList.contains('wardrobe-item')) {
                            setupItemId(node);
                        }
                    }
                });
            });
        });

        const wardrobeGrid = document.getElementById('wardrobe-items');
        if (wardrobeGrid) {
            observer.observe(wardrobeGrid, { childList: true, subtree: true });
        }
    }

    function setupItemId(wardrobeItem) {
        // Ensure item has proper ID for try-on mode
        if (isAvatarMode) {
            wardrobeItem.classList.add('try-on-mode');
        }
    }

    function showMessage(message, type = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.innerHTML = `<i class="bi bi-${getMessageIcon(type)}"></i> ${message}`;

        document.body.appendChild(messageDiv);

        setTimeout(() => {
            messageDiv.remove();
        }, 4000);
    }

    function getMessageIcon(type) {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-triangle',
            'info': 'info-circle',
            'warning': 'exclamation-circle'
        };
        return icons[type] || 'info-circle';
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeWardrobeIntegration);
    } else {
        initializeWardrobeIntegration();
    }

    // Also initialize after a delay to catch dynamically loaded content
    setTimeout(initializeWardrobeIntegration, 2000);

    console.log('📝 FIXED Wardrobe Avatar Integration script loaded');

})();