// avatar-clothing-init.js - Initializes the avatar clothing system and connections

document.addEventListener('DOMContentLoaded', function() {
    console.log('[Init] Initializing avatar clothing system');

    // Function to initialize the custom avatar clothing manager
    function initCustomAvatarClothing() {
        // Wait for both avatarClothingManager and global avatar/scene to be available
        const checkInterval = setInterval(() => {
            if (window.avatarClothingManager && window.avatar && window.scene) {
                clearInterval(checkInterval);
                console.log('[Init] Connecting avatar clothing manager to avatar and scene');

                // Set references
                window.avatarClothingManager.setReferences(window.avatar, window.scene);

                // Add global functions to ensure compatibility
                if (!window.applyClothing && window.avatarClothingManager.applyClothing) {
                    window.applyClothing = (item) => window.avatarClothingManager.applyClothing(item);
                }

                // Handle Test Custom Top button click
                const testCustomTopBtn = document.getElementById('test-custom-top-btn');
                if (testCustomTopBtn) {
                    testCustomTopBtn.addEventListener('click', async function() {
                        console.log('[Init] Test Custom Top button clicked');
                        try {
                            if (window.showLoading) window.showLoading('Loading test top...');

                            if (!window.avatar) {
                                console.log('[Init] No avatar found, loading female model first');
                                await loadAvatarModel({ gender: 'female' });
                            }

                            // Create a test clothing item
                            const testItem = {
                                id: 'test-top',
                                type: 'tops',
                                label: 'T-shirt/top',
                                file_path: '/static/models/clothing/top.glb'
                            };

                            // Apply the clothing
                            const result = await window.avatarClothingManager.applyClothing(testItem);

                            if (result) {
                                if (window.showMessage) window.showMessage('Test top applied successfully', 'success');
                            } else {
                                if (window.showMessage) window.showMessage('Failed to apply test top', 'error');
                            }
                        } catch (error) {
                            console.error('[Init] Error in test top button handler:', error);
                            if (window.showMessage) window.showMessage('Error applying test top', 'error');
                        } finally {
                            if (window.hideLoading) window.hideLoading();
                        }
                    });
                }

                // Set up try-on button for custom avatar
                const tryOnCustomBtn = document.getElementById('try-on-custom');
                if (tryOnCustomBtn) {
                    tryOnCustomBtn.addEventListener('click', async function() {
                        const selectedItems = document.querySelectorAll('#custom-avatar-section .wardrobe-item.selected');

                        if (selectedItems.length === 0) {
                            if (window.showMessage) window.showMessage('Please select items to try on', 'info');
                            return;
                        }

                        // Apply each selected item
                        for (const item of selectedItems) {
                            await window.avatarClothingManager.handleClothingItemClick(item);
                        }
                    });
                }

                console.log('[Init] Avatar clothing manager successfully initialized and connected');
            }
        }, 200);

        // Set timeout to avoid infinite waiting
        setTimeout(() => clearInterval(checkInterval), 10000);
    }

    // Handle avatar element visibility based on active tab
    function handleAvatarVisibility() {
        const tabs = document.querySelectorAll('.avatar-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                const tabName = this.dataset.tab;

                // Toggle avatar visibility based on selected tab
                if (tabName === 'custom') {
                    if (window.avatar) {
                        window.avatar.visible = true;
                    }
                    if (window.rpmManager && window.rpmManager.avatarModel) {
                        window.rpmManager.avatarModel.visible = false;
                    }
                } else if (tabName === 'rpm') {
                    if (window.avatar) {
                        window.avatar.visible = false;
                    }
                    if (window.rpmManager && window.rpmManager.avatarModel) {
                        window.rpmManager.avatarModel.visible = true;
                    }
                }
            });
        });
    }

    // Initialize the custom avatar clothing system
    initCustomAvatarClothing();

    // Setup avatar visibility handling
    handleAvatarVisibility();

    // Ensure clothing selection is updated for the active tab
    function updateClothingSelectionForActiveTab() {
        // Check which tab is active
        const customTabActive = document.querySelector('#custom-avatar-section.active') !== null;

        if (customTabActive) {
            // Make custom avatar visible
            if (window.avatar) {
                window.avatar.visible = true;
            }
            if (window.rpmManager && window.rpmManager.avatarModel) {
                window.rpmManager.avatarModel.visible = false;
            }
        } else {
            // Make RPM avatar visible
            if (window.avatar) {
                window.avatar.visible = false;
            }
            if (window.rpmManager && window.rpmManager.avatarModel) {
                window.rpmManager.avatarModel.visible = true;
            }
        }
    }

    // Call initially and then set interval to check regularly
    updateClothingSelectionForActiveTab();
    setInterval(updateClothingSelectionForActiveTab, 1000);
});