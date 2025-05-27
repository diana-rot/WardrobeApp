# triposg_config.py - Configuration for TripoSG integration

import os


class TripoSGConfig:
    """Configuration class for TripoSG 3D model generation"""

    # TripoSG API settings
    TRIPOSG_SPACE_ID = "VAST-AI/TripoSG"
    TRIPOSG_TIMEOUT = 300  # 5 minutes timeout

    # Generation parameters
    DEFAULT_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 7.0
    DEFAULT_TARGET_FACES = 100000
    DEFAULT_SIMPLIFY_MESH = True

    # File paths
    TEMP_3D_DIR = os.path.join('flaskapp', 'static', 'temp_3d_generation')
    GENERATED_MODELS_DIR = os.path.join('flaskapp', 'static', 'models', 'generated')

    # Task management
    MAX_CONCURRENT_GENERATIONS = 3
    TASK_CLEANUP_INTERVAL = 3600  # 1 hour
    MAX_TASK_AGE = 86400  # 24 hours

    # Model quality settings
    QUALITY_PRESETS = {
        'draft': {
            'num_inference_steps': 25,
            'guidance_scale': 5.0,
            'target_face_num': 50000,
            'simplify': True
        },
        'standard': {
            'num_inference_steps': 50,
            'guidance_scale': 7.0,
            'target_face_num': 100000,
            'simplify': True
        },
        'high': {
            'num_inference_steps': 75,
            'guidance_scale': 9.0,
            'target_face_num': 200000,
            'simplify': False
        },
        'ultra': {
            'num_inference_steps': 100,
            'guidance_scale': 10.0,
            'target_face_num': 500000,
            'simplify': False
        }
    }

    # Clothing type specific settings
    CLOTHING_TYPE_SETTINGS = {
        'T-shirt/top': {
            'preset': 'standard',
            'auto_generate': True,
            'priority': 'high'
        },
        'Dress': {
            'preset': 'high',
            'auto_generate': True,
            'priority': 'high'
        },
        'Trouser': {
            'preset': 'standard',
            'auto_generate': False,
            'priority': 'medium'
        },
        'Pullover': {
            'preset': 'standard',
            'auto_generate': True,
            'priority': 'medium'
        },
        'Coat': {
            'preset': 'high',
            'auto_generate': False,
            'priority': 'medium'
        },
        'Shirt': {
            'preset': 'standard',
            'auto_generate': True,
            'priority': 'high'
        },
        'Sandal': {
            'preset': 'draft',
            'auto_generate': False,
            'priority': 'low'
        },
        'Sneaker': {
            'preset': 'draft',
            'auto_generate': False,
            'priority': 'low'
        },
        'Bag': {
            'preset': 'standard',
            'auto_generate': False,
            'priority': 'low'
        },
        'Ankle boot': {
            'preset': 'draft',
            'auto_generate': False,
            'priority': 'low'
        }
    }

    @classmethod
    def get_settings_for_clothing_type(cls, clothing_type):
        """Get generation settings for specific clothing type"""
        return cls.CLOTHING_TYPE_SETTINGS.get(
            clothing_type,
            {
                'preset': 'standard',
                'auto_generate': False,
                'priority': 'low'
            }
        )

    @classmethod
    def get_quality_preset(cls, preset_name):
        """Get quality preset parameters"""
        return cls.QUALITY_PRESETS.get(preset_name, cls.QUALITY_PRESETS['standard'])

    @classmethod
    def should_auto_generate(cls, clothing_type):
        """Check if 3D model should be auto-generated for clothing type"""
        settings = cls.get_settings_for_clothing_type(clothing_type)
        return settings.get('auto_generate', False)


# Environment-specific overrides
if os.getenv('FLASK_ENV') == 'production':
    TripoSGConfig.MAX_CONCURRENT_GENERATIONS = 5
    TripoSGConfig.TRIPOSG_TIMEOUT = 600  # 10 minutes in production
elif os.getenv('FLASK_ENV') == 'development':
    TripoSGConfig.MAX_CONCURRENT_GENERATIONS = 2
    TripoSGConfig.DEFAULT_INFERENCE_STEPS = 25  # Faster for development