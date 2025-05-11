// RPM Avatar Position and Rendering Fixes
(function() {
    // Function to adjust the RPM avatar's size and position
    function adjustAvatarPosition(manager) {
        if (!manager || !manager.avatarModel) {
            console.log("RPM Avatar not found. Will try again when available.");
            return false;
        }

        const avatar = manager.avatarModel;
        
        // Scale and position adjustments
        avatar.scale.set(0.25, 0.25, 0.25); // Scale to 25% of original size (reduced from 40%)
        avatar.position.set(0, 1.2, 0); // Keep the same height position
        
        // Camera adjustments
        if (manager.camera) {
            manager.camera.position.set(0, 2.2, 2.8); // Keep the same camera position
        }
        
        // Controls adjustments
        if (manager.controls) {
            manager.controls.target.set(0, 1.7, 0); // Keep the same target position
            manager.controls.update();
        }
        
        console.log("RPM Avatar position and size adjusted successfully");
        return true;
    }

    // Create position controls UI
    function createPositionControls() {
        const avatarContainer = document.getElementById('avatar-container');
        if (!avatarContainer) return;
        
        const controlsPanel = document.createElement('div');
        controlsPanel.className = 'position-controls-panel';
        controlsPanel.innerHTML = `
            <div class="position-controls">
                <button class="position-btn" id="move-up-btn" title="Move Avatar Up">↑</button>
                <button class="position-btn" id="move-down-btn" title="Move Avatar Down">↓</button>
                <button class="position-btn" id="scale-down-btn" title="Make Avatar Smaller">-</button>
                <button class="position-btn" id="scale-up-btn" title="Make Avatar Larger">+</button>
                <button class="position-btn" id="reset-position-btn" title="Reset Position">⟲</button>
            </div>
        `;
        
        const style = document.createElement('style');
        style.textContent = `
            .position-controls-panel {
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 100;
            }
            
            .position-controls {
                display: flex;
                gap: 5px;
                background: rgba(255, 255, 255, 0.8);
                padding: 5px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            
            .position-btn {
                width: 30px;
                height: 30px;
                border-radius: 4px;
                border: 1px solid #ccc;
                background: white;
                cursor: pointer;
                font-size: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .position-btn:hover {
                background: #f0f0f0;
            }
        `;
        
        document.head.appendChild(style);
        avatarContainer.appendChild(controlsPanel);
        
        // Add event listeners
        document.getElementById('move-up-btn')?.addEventListener('click', () => {
            if (window.rpmManager?.avatarModel) {
                window.rpmManager.avatarModel.position.y += 0.1;
            }
        });
        
        document.getElementById('move-down-btn')?.addEventListener('click', () => {
            if (window.rpmManager?.avatarModel) {
                window.rpmManager.avatarModel.position.y -= 0.1;
            }
        });
        
        document.getElementById('scale-down-btn')?.addEventListener('click', () => {
            if (window.rpmManager?.avatarModel) {
                const currentScale = window.rpmManager.avatarModel.scale.x;
                window.rpmManager.avatarModel.scale.set(currentScale - 0.05, currentScale - 0.05, currentScale - 0.05);
            }
        });
        
        document.getElementById('scale-up-btn')?.addEventListener('click', () => {
            if (window.rpmManager?.avatarModel) {
                const currentScale = window.rpmManager.avatarModel.scale.x;
                window.rpmManager.avatarModel.scale.set(currentScale + 0.05, currentScale + 0.05, currentScale + 0.05);
            }
        });
        
        document.getElementById('reset-position-btn')?.addEventListener('click', () => {
            adjustAvatarPosition(window.rpmManager);
        });
    }

    // Initialize fixes when document is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Create position controls
        createPositionControls();
        
        // Override the original loadAvatar method
        if (window.RPMAvatarManager && RPMAvatarManager.prototype.loadAvatar) {
            const originalLoadAvatar = RPMAvatarManager.prototype.loadAvatar;
            
            RPMAvatarManager.prototype.loadAvatar = async function(url) {
                const result = await originalLoadAvatar.call(this, url);
                setTimeout(() => adjustAvatarPosition(this), 500);
                return result;
            };
        }

        // Override loadSavedAvatar method
        if (window.RPMAvatarManager && RPMAvatarManager.prototype.loadSavedAvatar) {
            const originalLoadSavedAvatar = RPMAvatarManager.prototype.loadSavedAvatar;
            
            RPMAvatarManager.prototype.loadSavedAvatar = async function() {
                const result = await originalLoadSavedAvatar.call(this);
                setTimeout(() => adjustAvatarPosition(this), 500);
                return result;
            };
        }

        // Override loadClothing to maintain position
        if (window.rpmManager && window.rpmManager.loadClothing) {
            const originalLoadClothing = window.rpmManager.loadClothing;
            
            window.rpmManager.loadClothing = async function(itemId, itemUrl, itemType) {
                const result = await originalLoadClothing.call(this, itemId, itemUrl, itemType);
                setTimeout(() => adjustAvatarPosition(this), 500);
                return result;
            };
        }
        
        // Apply fixes to existing avatar
        if (window.rpmManager?.avatarModel) {
            adjustAvatarPosition(window.rpmManager);
        } else {
            // Set up watcher for avatar loading
            let checkCount = 0;
            const maxChecks = 20;
            
            const checkInterval = setInterval(() => {
                if (window.rpmManager?.avatarModel) {
                    adjustAvatarPosition(window.rpmManager);
                    clearInterval(checkInterval);
                } else if (++checkCount >= maxChecks) {
                    clearInterval(checkInterval);
                }
            }, 500);
        }
        
        // Apply fixes when loading avatars
        const avatarButtons = [
            'load-saved-rpm',
            'load-rpm-url',
            'load-top-rpm'
        ];
        
        avatarButtons.forEach(buttonId => {
            const button = document.getElementById(buttonId);
            if (button) {
                button.addEventListener('click', () => {
                    setTimeout(() => adjustAvatarPosition(window.rpmManager), 1000);
                });
            }
        });
    });
})(); 