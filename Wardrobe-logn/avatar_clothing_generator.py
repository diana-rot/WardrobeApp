import os
import json
import numpy as np
from PIL import Image
import cv2
from flask import current_app
import trimesh
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Accessor, BufferView, Buffer, Material, Texture, Image as GLTFImage, TextureInfo
from PIL import Image as PILImage
import io
from bson import ObjectId
from pymongo import MongoClient
from gridfs import GridFS

class AvatarClothingGenerator:
    def __init__(self, wardrobe_path, output_path, db_client=None):
        """Initialize the AvatarClothingGenerator.
        
        Args:
            wardrobe_path (str): Path to the wardrobe images directory
            output_path (str): Path to save generated assets
            db_client (MongoClient, optional): MongoDB client for GridFS storage
        """
        self.wardrobe_path = wardrobe_path
        self.output_path = output_path
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(output_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'textures'), exist_ok=True)
        
        # Initialize GridFS if db_client is provided
        self.fs = None
        if db_client:
            db = db_client.user_login_system_test
            self.fs = GridFS(db)
    
    def process_wardrobe_image(self, image_path):
        """Process a wardrobe image to create a texture with transparency.
        
        Args:
            image_path (str): Path to the wardrobe image
            
        Returns:
            tuple: (numpy.ndarray, bytes) Processed image array and PNG bytes
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create mask for white/light backgrounds
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
        
        # Invert mask to get the clothing
        mask = cv2.bitwise_not(mask_white)
        
        # Clean up mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (the clothing item)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Create RGBA image with transparency
        alpha = mask.astype(np.uint8)
        img_resized = cv2.resize(img, (1024, 1024))
        alpha_resized = cv2.resize(alpha, (1024, 1024))
        rgba = np.dstack((img_resized, alpha_resized))
        
        # Convert to PNG bytes
        pil_image = Image.fromarray(rgba)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return rgba, img_byte_arr
    
    def save_texture(self, image_array, output_path, item_id=None):
        """Save the processed image as a PNG file with transparency.
        
        Args:
            image_array (numpy.ndarray): RGBA image array
            output_path (str): Path to save the texture
            item_id (str, optional): ID of the wardrobe item for GridFS storage
            
        Returns:
            str: Path or GridFS ID of the saved texture
        """
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        
        if self.fs and item_id:
            # Save to GridFS
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Store in GridFS with metadata
            texture_id = self.fs.put(
                img_byte_arr.getvalue(),
                filename=f'texture_{item_id}.png',
                metadata={
                    'item_id': item_id,
                    'type': 'texture',
                    'format': 'png'
                }
            )
            return str(texture_id)
        else:
            # Save to file system
            image.save(output_path, 'PNG')
            return output_path
    
    def get_texture(self, texture_id):
        """Retrieve a texture from GridFS or file system.
        
        Args:
            texture_id (str): GridFS ID or file path of the texture
            
        Returns:
            bytes: The texture data
        """
        if self.fs:
            try:
                # Try to get from GridFS
                grid_out = self.fs.get(ObjectId(texture_id))
                return grid_out.read()
            except:
                return None
        else:
            # Read from file system
            try:
                with open(texture_id, 'rb') as f:
                    return f.read()
            except:
                return None

    def create_basic_top_mesh(self):
        """Create a basic top mesh for the avatar.
        
        Returns:
            trimesh.Trimesh: The created mesh
        """
        # Create a more detailed top mesh
        vertices = np.array([
            # Front panel
            [-0.4, 0.0, 0.05],    # Bottom left
            [0.4, 0.0, 0.05],     # Bottom right
            [0.4, 0.7, 0.05],     # Top right
            [-0.4, 0.7, 0.05],    # Top left
            
            # Back panel
            [-0.4, 0.0, -0.05],   # Bottom left
            [0.4, 0.0, -0.05],    # Bottom right
            [0.4, 0.7, -0.05],    # Top right
            [-0.4, 0.7, -0.05],   # Top left
            
            # Shoulders
            [-0.5, 0.6, 0.05],    # Left shoulder front
            [0.5, 0.6, 0.05],     # Right shoulder front
            [-0.5, 0.6, -0.05],   # Left shoulder back
            [0.5, 0.6, -0.05],    # Right shoulder back
            
            # Sleeves
            [-0.6, 0.4, 0.05],    # Left sleeve front
            [0.6, 0.4, 0.05],     # Right sleeve front
            [-0.6, 0.4, -0.05],   # Left sleeve back
            [0.6, 0.4, -0.05],    # Right sleeve back
        ])
        
        faces = np.array([
            # Front panel
            [0, 1, 2], [0, 2, 3],
            # Back panel
            [4, 5, 6], [4, 6, 7],
            # Left side
            [0, 3, 7], [0, 7, 4],
            # Right side
            [1, 5, 6], [1, 6, 2],
            # Left shoulder front
            [3, 8, 12], [3, 12, 0],
            # Right shoulder front
            [2, 9, 13], [2, 13, 1],
            # Left shoulder back
            [7, 10, 14], [7, 14, 4],
            # Right shoulder back
            [6, 11, 15], [6, 15, 5],
            # Shoulder tops
            [8, 9, 11], [8, 11, 10],
            # Left sleeve
            [8, 12, 14], [8, 14, 10],
            # Right sleeve
            [9, 13, 15], [9, 15, 11],
        ])
        
        # Create UV coordinates for better texture mapping
        uv = np.array([
            # Front panel UVs
            [0.2, 0.8], [0.8, 0.8],     # Bottom edge
            [0.8, 0.2], [0.2, 0.2],     # Top edge
            # Back panel UVs
            [0.2, 0.8], [0.8, 0.8],     # Bottom edge
            [0.8, 0.2], [0.2, 0.2],     # Top edge
            # Shoulder UVs
            [0.1, 0.3], [0.9, 0.3],     # Front shoulders
            [0.1, 0.3], [0.9, 0.3],     # Back shoulders
            # Sleeve UVs
            [0.0, 0.4], [1.0, 0.4],     # Front sleeves
            [0.0, 0.4], [1.0, 0.4],     # Back sleeves
        ])
        
        # Create the mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.uv = uv
        
        return mesh
    
    def create_gltf(self, mesh, texture_path, output_path):
        """Create a GLTF file from a mesh and texture.
        
        Args:
            mesh (trimesh.Trimesh): The mesh to export
            texture_path (str): Path to the texture image
            output_path (str): Path to save the GLTF file
        """
        # Create a new GLTF2 object
        gltf = GLTF2()
        
        # Add a scene
        scene = Scene(nodes=[0])
        gltf.scenes.append(scene)
        gltf.scene = 0
        
        # Add a node
        node = Node(mesh=0)
        gltf.nodes.append(node)
        
        # Convert mesh to GLTF format
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint16)
        
        # Read texture image
        with open(texture_path, 'rb') as f:
            image_data = f.read()
        
        # Create buffer data
        vertices_data = vertices.tobytes()
        indices_data = faces.tobytes()
        
        # Combine all data into a single buffer
        buffer_data = vertices_data + indices_data + image_data
        
        # Add buffer
        buffer = Buffer(byteLength=len(buffer_data))
        gltf.buffers.append(buffer)
        
        # Add buffer views
        # Vertices buffer view
        vertices_view = BufferView(
            buffer=0,
            byteOffset=0,
            byteLength=len(vertices_data),
            target=34962  # ARRAY_BUFFER
        )
        gltf.bufferViews.append(vertices_view)
        
        # Indices buffer view
        indices_view = BufferView(
            buffer=0,
            byteOffset=len(vertices_data),
            byteLength=len(indices_data),
            target=34963  # ELEMENT_ARRAY_BUFFER
        )
        gltf.bufferViews.append(indices_view)
        
        # Image buffer view
        image_view = BufferView(
            buffer=0,
            byteOffset=len(vertices_data) + len(indices_data),
            byteLength=len(image_data)
        )
        gltf.bufferViews.append(image_view)
        
        # Add accessors
        vertices_accessor = Accessor(
            bufferView=0,
            componentType=5126,  # FLOAT
            count=len(vertices),
            type="VEC3",
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist()
        )
        gltf.accessors.append(vertices_accessor)
        
        indices_accessor = Accessor(
            bufferView=1,
            componentType=5123,  # UNSIGNED_SHORT
            count=len(faces.flatten()),
            type="SCALAR"
        )
        gltf.accessors.append(indices_accessor)
        
        # Add material with alpha mode
        material = Material(
            pbrMetallicRoughness={
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0
            },
            alphaMode="MASK",
            alphaCutoff=0.5,
            doubleSided=True
        )
        gltf.materials.append(material)
        
        # Add texture with sampler
        texture = Texture(
            source=0,
            sampler=0
        )
        gltf.textures.append(texture)
        
        # Add image
        image = GLTFImage(
            bufferView=2,
            mimeType="image/png"
        )
        gltf.images.append(image)
        
        # Add UV coordinates accessor
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uv_data = mesh.visual.uv.astype(np.float32)
            uv_bytes = uv_data.tobytes()
            
            # UV buffer view
            uv_view = BufferView(
                buffer=0,
                byteOffset=len(vertices_data) + len(indices_data) + len(image_data),
                byteLength=len(uv_bytes),
                target=34962  # ARRAY_BUFFER
            )
            gltf.bufferViews.append(uv_view)
            
            # UV accessor
            uv_accessor = Accessor(
                bufferView=3,
                componentType=5126,  # FLOAT
                count=len(uv_data),
                type="VEC2",
                max=uv_data.max(axis=0).tolist(),
                min=uv_data.min(axis=0).tolist()
            )
            gltf.accessors.append(uv_accessor)
            
            # Update buffer data to include UVs
            buffer_data += uv_bytes
            buffer.byteLength = len(buffer_data)
        
        # Add primitive with UV coordinates
        primitive = Primitive(
            attributes={
                "POSITION": 0,
                "TEXCOORD_0": 3  # Reference to UV accessor
            },
            indices=1,
            material=0
        )
        
        # Add mesh
        mesh = Mesh(primitives=[primitive])
        gltf.meshes.append(mesh)
        
        # Save the GLTF file
        gltf.set_binary_blob(buffer_data)
        gltf.save(output_path)

    def generate_top_from_wardrobe(self, wardrobe_item_path, item_id=None):
        """Generate a top 3D model from a wardrobe item"""
        try:
            # Process the wardrobe image
            texture_array, texture_bytes = self.process_wardrobe_image(wardrobe_item_path)
            if texture_array is None:
                return False
            
            # Save the processed texture
            if item_id:
                texture_id = self.save_texture(texture_array, None, item_id)
                texture_path = os.path.join(self.output_path, 'textures', f'temp_{item_id}.png')
                with open(texture_path, 'wb') as f:
                    f.write(texture_bytes)
            else:
                texture_path = os.path.join(self.output_path, 'textures', 'top_texture.png')
                self.save_texture(texture_array, texture_path)
                texture_id = texture_path
            
            # Create the basic top mesh
            mesh = self.create_basic_top_mesh()
            
            # Create the GLTF file
            output_path = os.path.join(self.output_path, 'models', f'{"top_" + item_id if item_id else "top"}.glb')
            self.create_gltf(mesh, texture_path, output_path)
            
            # Clean up temporary texture file if using GridFS
            if item_id and os.path.exists(texture_path):
                os.remove(texture_path)
            
            return True, texture_id, output_path
            
        except Exception as e:
            print(f"Error generating top: {str(e)}")
            return False, None, None

    def generate_from_image(self, image_path, category):
        """Generate a 3D model from a wardrobe image.
        
        Args:
            image_path (str): Path to the wardrobe image
            category (str): Category of clothing (e.g., 'T-shirt/top', 'Dress', etc.)
            
        Returns:
            dict: Model information including URL and metadata
        """
        try:
            # Load and process the image
            img = PILImage.open(os.path.join(self.wardrobe_path, image_path))
            
            # Create texture from the image
            texture_path = self._create_texture(img, category)
            
            # Generate 3D model based on category
            model_path = self._generate_3d_model(category, texture_path)
            
            if not model_path:
                raise Exception("Failed to generate 3D model")

            return {
                'success': True,
                'model_url': f'/static/models/generated/{os.path.basename(model_path)}',
                'category': category
            }

        except Exception as e:
            print(f"Error generating 3D clothing: {str(e)}")
            return None

    def _create_texture(self, img, category):
        """Create a texture map from the wardrobe image.
        
        Args:
            img (PIL.Image): Source image
            category (str): Clothing category
            
        Returns:
            str: Path to the generated texture
        """
        # Resize image to power of 2 dimensions
        size = (512, 512)
        img = img.resize(size, PILImage.LANCZOS)
        
        # Create texture filename
        texture_name = f"texture_{category}_{ObjectId()}.png"
        texture_path = os.path.join(self.output_path, 'textures', texture_name)
        
        # Save texture
        img.save(texture_path, 'PNG')
        
        return texture_path

    def _generate_3d_model(self, category, texture_path):
        """Generate a 3D model for the clothing item.
        
        Args:
            category (str): Clothing category
            texture_path (str): Path to the texture file
            
        Returns:
            str: Path to the generated model
        """
        # Load base model template based on category
        base_model = self._get_base_model(category)
        if not base_model:
            return None

        # Create model filename
        model_name = f"model_{category}_{ObjectId()}.glb"
        model_path = os.path.join(self.output_path, 'models', model_name)

        try:
            # Apply texture to model
            textured_model = self._apply_texture_to_model(base_model, texture_path)
            
            # Save the model
            textured_model.export(model_path)
            
            return model_path

        except Exception as e:
            print(f"Error generating 3D model: {str(e)}")
            return None

    def _get_base_model(self, category):
        """Get the base 3D model template for a clothing category.
        
        Args:
            category (str): Clothing category
            
        Returns:
            trimesh.Scene: Base model template
        """
        # Map categories to base model templates
        category_templates = {
            'T-shirt/top': 'tshirt_template.glb',
            'Trouser': 'pants_template.glb',
            'Pullover': 'pullover_template.glb',
            'Dress': 'dress_template.glb',
            'Coat': 'coat_template.glb',
            'Sandal': 'sandal_template.glb',
            'Shirt': 'shirt_template.glb',
            'Sneaker': 'sneaker_template.glb',
            'Bag': 'bag_template.glb',
            'Ankle boot': 'boot_template.glb'
        }
        
        template_file = category_templates.get(category)
        if not template_file:
            return None
            
        template_path = os.path.join('flaskapp/static/models/templates', template_file)
        
        try:
            return trimesh.load(template_path)
        except Exception as e:
            print(f"Error loading template model: {str(e)}")
            return None

    def _apply_texture_to_model(self, model, texture_path):
        """Apply a texture to a 3D model.
        
        Args:
            model (trimesh.Scene): The 3D model
            texture_path (str): Path to the texture file
            
        Returns:
            trimesh.Scene: Textured model
        """
        try:
            # Load the texture image
            texture = PILImage.open(texture_path)
            
            # Create material with the texture
            material = trimesh.visual.material.SimpleMaterial(
                image=texture,
                diffuse=[255, 255, 255, 255],
                ambient=[100, 100, 100, 255],
                specular=[100, 100, 100, 255]
            )
            
            # Apply material to all mesh faces
            for mesh in model.geometry.values():
                mesh.visual = trimesh.visual.TextureVisuals(
                    material=material,
                    uv=mesh.visual.uv
                )
            
            return model

        except Exception as e:
            print(f"Error applying texture: {str(e)}")
            return model  # Return untextured model as fallback