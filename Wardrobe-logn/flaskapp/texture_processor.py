import cv2
import numpy as np
from PIL import Image
from rembg import remove
import os
import concurrent.futures
from skimage import exposure, color
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TextureProcessor')


class OptimizedTextureProcessor:
    def __init__(self, uv_maps_file=None):
        """
        Initialize texture processor with configurable UV maps

        Args:
            uv_maps_file (str, optional): Path to a JSON file containing UV maps.
                                         If None, default UV maps will be used.
        """
        # Set constants for texture resolution
        self.TEXTURE_SIZE = 2048  # Higher resolution for AI applications
        self.NORMAL_SIZE = 1024  # Normal maps can be smaller resolution

        # Load UV maps from file or use defaults
        self.uv_maps = self._load_uv_maps(uv_maps_file)

        # Create a memory cache for processed images
        self._cache = {}

        logger.info("TextureProcessor initialized with texture size: %d", self.TEXTURE_SIZE)

    def _load_uv_maps(self, uv_maps_file):
        """Load UV maps from file or use defaults"""
        default_maps = {
            'T-shirt/top': {
                'front': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (0.0, 0.6)],
                'back': [(0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)]
            },
            'Pullover': {
                'front': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (0.0, 0.6)],
                'back': [(0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)]
            },
            'Dress': {
                'front': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
                'back': [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
            },
            'Trouser': {
                'front_left': [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
                'front_right': [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
            },
            'Coat': {
                'front': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (0.0, 0.6)],
                'back': [(0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)]
            },
            'Sandal': {
                'top': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
                'bottom': [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
            },
            'Shirt': {
                'front': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (0.0, 0.6)],
                'back': [(0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)]
            },
            'Sneaker': {
                'top': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
                'side': [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
            },
            'Bag': {
                'front': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
                'back': [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
            },
            'Ankle boot': {
                'top': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
                'side': [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
            }
        }

        if uv_maps_file and os.path.exists(uv_maps_file):
            try:
                with open(uv_maps_file, 'r') as f:
                    loaded_maps = json.load(f)
                    # Validate and merge with defaults for any missing items
                    for item_type, uv_map in default_maps.items():
                        if item_type not in loaded_maps:
                            loaded_maps[item_type] = uv_map
                    return loaded_maps
            except Exception as e:
                logger.error(f"Error loading UV maps file: {str(e)}, using defaults")
                return default_maps
        else:
            return default_maps

    def remove_background(self, image_path, cache_key=None):
        """
        Remove background from clothing image using rembg and create transparent PNG

        Args:
            image_path (str): Path to input image
            cache_key (str, optional): Key for caching results

        Returns:
            str: Path to the processed image with background removed
        """
        # Check cache first
        if cache_key and cache_key in self._cache:
            logger.debug(f"Using cached background removal for {cache_key}")
            return self._cache[cache_key]

        try:
            # Create output path
            base_path, ext = os.path.splitext(image_path)
            output_path = f"{base_path}_nobg.png"

            # Check if output already exists
            if os.path.exists(output_path):
                logger.debug(f"Using existing background-removed image: {output_path}")
                if cache_key:
                    self._cache[cache_key] = output_path
                return output_path

            # Load image
            input_image = Image.open(image_path)

            # Remove background with rembg
            output_image = remove(input_image)

            # Save output
            output_image.save(output_path)

            # Cache result
            if cache_key:
                self._cache[cache_key] = output_path

            logger.info(f"Background removed successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error in background removal: {str(e)}")
            return None

    def auto_segment_clothing(self, image_path):
        """
        Automatically segment the clothing from the image

        Args:
            image_path (str): Path to input image

        Returns:
            tuple: (mask, contours) Segmentation mask and contours
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get largest contour (assuming it's the clothing)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create mask from largest contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)

            return mask, [largest_contour]

        except Exception as e:
            logger.error(f"Error in clothing segmentation: {str(e)}")
            return None, None

    def enhance_texture(self, image):
        """
        Enhance texture details for better visualization

        Args:
            image (ndarray): Input image

        Returns:
            ndarray: Enhanced image
        """
        try:
            # Make a copy of the image
            enhanced = image.copy()

            # Convert to appropriate format for processing
            if enhanced.shape[2] == 4:  # Has alpha channel
                # Split into RGB and alpha
                rgb = enhanced[:, :, :3]
                alpha = enhanced[:, :, 3]

                # Convert to LAB color space
                lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)

                # Split LAB channels
                l, a, b = cv2.split(lab)

                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)

                # Merge channels back
                enhanced_lab = cv2.merge((cl, a, b))

                # Convert back to RGB
                enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                # Merge with alpha channel
                enhanced = cv2.merge((enhanced_rgb[:, :, 0],
                                      enhanced_rgb[:, :, 1],
                                      enhanced_rgb[:, :, 2],
                                      alpha))
            else:
                # Convert to LAB color space
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

                # Split LAB channels
                l, a, b = cv2.split(lab)

                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)

                # Merge channels back
                enhanced_lab = cv2.merge((cl, a, b))

                # Convert back to RGB
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Apply subtle sharpening
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]]) / 9
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            return enhanced

        except Exception as e:
            logger.error(f"Error in texture enhancement: {str(e)}")
            return image  # Return original if enhancement fails

    def warp_image_to_uv(self, image, uv_map, texture_size=None):
        """
        Warp the image to fit the UV map for the 3D model

        Args:
            image (ndarray): Input image
            uv_map (dict): UV mapping coordinates
            texture_size (int, optional): Size of the output texture

        Returns:
            ndarray: Warped image for UV mapping
        """
        if texture_size is None:
            texture_size = self.TEXTURE_SIZE

        try:
            # Get image dimensions
            if image is None:
                raise ValueError("Input image is None")

            if len(image.shape) < 2:
                raise ValueError(f"Invalid image shape: {image.shape}")

            height, width = image.shape[:2]

            # Create output image with alpha channel
            channels = 4 if image.shape[2] == 4 else 3
            output = np.zeros((texture_size, texture_size, channels), dtype=np.uint8)

            # Process each UV section
            for section_name, coords in uv_map.items():
                logger.debug(f"Processing section: {section_name}")

                # Convert normalized coordinates to pixel coordinates
                src_points = np.array([
                    [int(coords[0][0] * width), int(coords[0][1] * height)],
                    [int(coords[1][0] * width), int(coords[1][1] * height)],
                    [int(coords[2][0] * width), int(coords[2][1] * height)],
                    [int(coords[3][0] * width), int(coords[3][1] * height)]
                ], dtype=np.float32)

                # Calculate target coordinates in texture space
                # Adjust based on UV section name and texture layout
                if 'front' in section_name:
                    dst_y_start = 0
                    dst_y_end = texture_size // 2
                elif 'back' in section_name:
                    dst_y_start = texture_size // 2
                    dst_y_end = texture_size
                else:
                    # For other sections, use full height
                    dst_y_start = 0
                    dst_y_end = texture_size

                if 'left' in section_name:
                    dst_x_start = 0
                    dst_x_end = texture_size // 2
                elif 'right' in section_name:
                    dst_x_start = texture_size // 2
                    dst_x_end = texture_size
                else:
                    # For other sections, use full width
                    dst_x_start = 0
                    dst_x_end = texture_size

                # Define destination points in texture space
                dst_points = np.array([
                    [dst_x_start, dst_y_start],
                    [dst_x_end, dst_y_start],
                    [dst_x_end, dst_y_end],
                    [dst_x_start, dst_y_end]
                ], dtype=np.float32)

                # Calculate perspective transform
                M = cv2.getPerspectiveTransform(src_points, dst_points)

                # Apply perspective transform
                try:
                    section_warped = cv2.warpPerspective(
                        image,
                        M,
                        (texture_size, texture_size),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_TRANSPARENT
                    )

                    # Blend with output
                    if channels == 4:
                        # For images with alpha channel, use alpha for blending
                        alpha = section_warped[:, :, 3] / 255.0
                        for c in range(3):
                            output[:, :, c] = (
                                    output[:, :, c] * (1 - alpha) +
                                    section_warped[:, :, c] * alpha
                            ).astype(np.uint8)
                        # Keep maximum alpha
                        output[:, :, 3] = np.maximum(output[:, :, 3], section_warped[:, :, 3])
                    else:
                        # For images without alpha, use addWeighted
                        mask = np.zeros((texture_size, texture_size), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, dst_points.astype(np.int32), 255)
                        mask3 = cv2.merge([mask, mask, mask])

                        # Apply mask to section
                        masked_section = cv2.bitwise_and(section_warped, mask3)

                        # Apply to output
                        masked_output = cv2.bitwise_and(output, cv2.bitwise_not(mask3))
                        output = cv2.add(masked_output, masked_section)
                except Exception as e:
                    logger.error(f"Error warping section {section_name}: {str(e)}")

            # Enhance texture quality
            output = self.enhance_texture(output)

            return output

        except Exception as e:
            logger.error(f"Error in UV warping: {str(e)}")
            return None

    def generate_normal_map(self, texture_path, output_size=None):
        """
        Generate high-quality normal map from texture

        Args:
            texture_path (str): Path to texture image
            output_size (int, optional): Size of output normal map

        Returns:
            str: Path to generated normal map
        """
        if output_size is None:
            output_size = self.NORMAL_SIZE

        try:
            # Load texture image
            texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
            if texture is None:
                raise ValueError(f"Could not load texture: {texture_path}")

            # Create normal map output path
            normal_map_path = texture_path.replace('_texture.png', '_normal.png')

            # Check if normal map already exists
            if os.path.exists(normal_map_path):
                logger.debug(f"Using existing normal map: {normal_map_path}")
                return normal_map_path

            # Convert to grayscale
            if texture.shape[2] == 4:
                # For images with alpha, use weighted RGB for better detail
                r, g, b, a = cv2.split(texture)
                gray = cv2.addWeighted(
                    cv2.addWeighted(r, 0.299, g, 0.587, 0),
                    1.0, b, 0.114, 0
                )
                # Apply alpha mask
                gray = cv2.bitwise_and(gray, a)
            else:
                gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)

            # Resize for consistency
            gray = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)

            # Apply bilateral filter for edge-preserving smoothing
            smooth = cv2.bilateralFilter(gray, 9, 75, 75)

            # Apply detail enhancement
            detail = cv2.detailEnhance(cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR), sigma_s=10, sigma_r=0.15)
            detail_gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)

            # Generate gradients for normal map
            sobelx = cv2.Sobel(detail_gray, cv2.CV_32F, 1, 0, ksize=5)
            sobely = cv2.Sobel(detail_gray, cv2.CV_32F, 0, 1, ksize=5)

            # Normalize gradients
            sobelx = cv2.normalize(sobelx, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            sobely = cv2.normalize(sobely, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

            # Create normal map (RGB)
            normal_map = np.zeros((output_size, output_size, 3), dtype=np.uint8)
            normal_map[..., 0] = sobelx * 127.5 + 127.5  # Red channel (X)
            normal_map[..., 1] = sobely * 127.5 + 127.5  # Green channel (Y)
            normal_map[..., 2] = 255  # Blue channel (Z always points up)

            # Save normal map with compression
            cv2.imwrite(normal_map_path, normal_map, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            logger.info(f"Normal map generated: {normal_map_path}")
            return normal_map_path

        except Exception as e:
            logger.error(f"Error in normal map generation: {str(e)}")
            return None

    def generate_metallic_roughness_map(self, texture_path, material_type='fabric'):
        """
        Generate metallic-roughness map for PBR materials

        Args:
            texture_path (str): Path to texture image
            material_type (str): Type of material (fabric, leather, etc.)

        Returns:
            str: Path to generated metallic-roughness map
        """
        try:
            # Create output path
            mr_map_path = texture_path.replace('_texture.png', '_roughness.png')

            # Check if map already exists
            if os.path.exists(mr_map_path):
                logger.debug(f"Using existing roughness map: {mr_map_path}")
                return mr_map_path

            # Load texture
            texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
            if texture is None:
                raise ValueError(f"Could not load texture: {texture_path}")

            # Convert to grayscale
            if texture.shape[2] == 4:
                # For images with alpha
                r, g, b, a = cv2.split(texture)
                gray = cv2.addWeighted(
                    cv2.addWeighted(r, 0.299, g, 0.587, 0),
                    1.0, b, 0.114, 0
                )
                # Apply alpha mask
                gray = cv2.bitwise_and(gray, a)
            else:
                gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)

            # Create roughness map based on material type
            if material_type == 'leather':
                # Leather is smooth with some variation
                base_roughness = 0.3
                variation = 0.1
            elif material_type == 'denim':
                # Denim is rough with high variation
                base_roughness = 0.8
                variation = 0.15
            else:  # Default fabric
                # Regular fabric has medium roughness
                base_roughness = 0.6
                variation = 0.2

            # Create roughness map by analyzing local variance
            kernel_size = 5
            local_var = cv2.GaussianBlur(
                cv2.Laplacian(gray, cv2.CV_32F),
                (kernel_size, kernel_size), 0
            )
            local_var = cv2.normalize(local_var, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

            # Scale to desired roughness range
            roughness = (base_roughness + local_var * variation) * 255
            roughness = np.clip(roughness, 0, 255).astype(np.uint8)

            # Create metallic channel (most fabrics are not metallic)
            metallic = np.zeros_like(roughness)

            # Create combined map (R=metallic, G=roughness)
            mr_map = np.zeros((roughness.shape[0], roughness.shape[1], 3), dtype=np.uint8)
            mr_map[..., 0] = metallic
            mr_map[..., 1] = roughness
            mr_map[..., 2] = 0  # Unused channel

            # Save to file
            cv2.imwrite(mr_map_path, mr_map, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            logger.info(f"Metallic-roughness map generated: {mr_map_path}")
            return mr_map_path

        except Exception as e:
            logger.error(f"Error generating metallic-roughness map: {str(e)}")
            return None

    def process_clothing_texture(self, image_path, clothing_type, material_type='fabric'):
        """
        Process a clothing image through the optimized pipeline

        Args:
            image_path (str): Path to input image
            clothing_type (str): Type of clothing (T-shirt, Dress, etc.)
            material_type (str): Material type for PBR properties

        Returns:
            dict: Paths to generated textures
        """
        logger.info(f"Processing texture for {clothing_type}: {image_path}")

        try:
            # Get base filename for caching
            base_name = os.path.basename(image_path)
            cache_key = f"{base_name}_{clothing_type}"

            # 1. Remove background
            start_time = cv2.getTickCount()
            nobg_path = self.remove_background(image_path, cache_key)
            if not nobg_path:
                raise Exception("Background removal failed")

            bg_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            logger.debug(f"Background removal time: {bg_time:.2f}s")

            # 2. Get UV map for clothing type
            clothing_key = clothing_type
            if '/' in clothing_key:
                clothing_key = clothing_type.split('/')[0]

            uv_map = self.uv_maps.get(clothing_type)
            if not uv_map:
                # Try with more generic key
                uv_map = self.uv_maps.get(clothing_key)

            if not uv_map:
                logger.warning(f"No UV map found for {clothing_type}, using default")
                # Use T-shirt map as default
                uv_map = self.uv_maps.get('T-shirt/top')

            # 3. Load the segmented image
            start_time = cv2.getTickCount()
            clothing_image = cv2.imread(nobg_path, cv2.IMREAD_UNCHANGED)
            if clothing_image is None:
                raise Exception(f"Failed to load segmented image: {nobg_path}")

            # 4. Warp image to UV map
            warped_texture = self.warp_image_to_uv(clothing_image, uv_map)
            if warped_texture is None:
                raise Exception("UV warping failed")

            warp_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            logger.debug(f"UV warping time: {warp_time:.2f}s")

            # 5. Save the texture
            texture_path = nobg_path.replace('_nobg.png', '_texture.png')
            cv2.imwrite(texture_path, warped_texture, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Process maps in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # 6. Generate normal map
                normal_map_future = executor.submit(
                    self.generate_normal_map, texture_path
                )

                # 7. Generate metallic-roughness map
                mr_map_future = executor.submit(
                    self.generate_metallic_roughness_map, texture_path, material_type
                )

                # Wait for all tasks to complete
                normal_map_path = normal_map_future.result()
                mr_map_path = mr_map_future.result()

            # 8. Determine appropriate 3D model based on clothing type
            model_map = {
                'T-shirt/top': 'tshirt.glb',
                'Pullover': 'pullover.glb',
                'Coat': 'coat.glb',
                'Shirt': 'shirt.glb',
                'Trouser': 'trouser.glb',
                'Dress': 'dress.glb',
                'Sandal': 'sandal.glb',
                'Sneaker': 'sneaker.glb',
                'Bag': 'bag.glb',
                'Ankle boot': 'boot.glb'
            }

            model_file = model_map.get(clothing_type, 'tshirt.glb')
            model_path = f"/static/models/clothing/{model_file}"

            # 9. Return all paths
            return {
                'texture_path': texture_path,
                'normal_map_path': normal_map_path,
                'roughness_map_path': mr_map_path,
                'model_path': model_path
            }

        except Exception as e:
            logger.error(f"Error in texture processing pipeline: {str(e)}")
            return None

    def batch_process_directory(self, input_dir, output_dir=None, clothing_type='T-shirt/top'):
        """
        Process all images in a directory

        Args:
            input_dir (str): Directory containing input images
            output_dir (str, optional): Directory for output textures
            clothing_type (str): Type of clothing

        Returns:
            list: Paths to all processed textures
        """
        if output_dir is None:
            output_dir = input_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get all image files
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))

        logger.info(f"Found {len(image_files)} images to process")

        # Process all images in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {}

            for image_file in image_files:
                # Create output path
                base_name = os.path.basename(image_file)
                output_path = os.path.join(output_dir, base_name)

                # Submit task
                future = executor.submit(
                    self.process_clothing_texture,
                    image_file, clothing_type
                )

                future_to_file[future] = output_path

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")

        return results