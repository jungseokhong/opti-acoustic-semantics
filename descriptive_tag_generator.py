from params_proto import PrefixProto, Proto


class tag_generator(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o-2024-05-13" # "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    temperature = 1.0
    max_tokens = 50
    # Use the cropped images for reference.
    system_prompt = """
Step 1: Are there multiple <object> in the box <tag number>?
Example:
 tag 0: yes, there are <number of <object>>

Step 2: Identify the features of the object inside the bounding box for each tag number. If there are multiple objects of the same category in the box, list each object with its unique color separately.
 1. Color: What is the primary color of the <object>? If there are multiple objects of the same category, list each object with its unique color separately.
 2. Unique features: What features most effectively distinguish this <object> from similar objects?
Example:
 <object>: 
 Tag 0 1. green, 2. green, with dot pattern
 tag 0 1. brown color, 2. brown color, with a handle 
  
Step 3: Create a descriptive tag for each <object> based on your answers. Use the format: tag_<tag_number> = ['<color> <object>', '<unique features> <object>' ]
Example:
 tag_0 = ['green <object>', 'green <object> with a handle']
 tag_0 = ['brown <object>', 'brown <object> with a handle']
 
"""
