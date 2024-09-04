from params_proto import PrefixProto, Proto

class tag_generator(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    temperature = 1.0
    max_tokens = 50
    system_prompt = """
Step 1: AIdentify the features of the object inside the bounding box for each tag number. Use the cropped images for reference.
 1. Color: What is the primary color of the <object>?
 2. Shape: What shape or structural features does the <object> have?
 3. Distinguishing Features: What features most effectively distinguish this <object> from similar objects?
Example:
 <object>: 1. brown color, 2. rectangle shape, 3. brown color, with a handle 

Step 2: Create a descriptive tag for each <object> based on your answers. Use the format: tag_<tab_number> = [].
Example:
 tag_<tag_number> = ['brown <object>', 'rectangle shape <object>', 'brown <object> with a handle']
"""