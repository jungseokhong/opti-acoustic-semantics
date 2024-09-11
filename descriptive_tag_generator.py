from params_proto import PrefixProto, Proto


class tag_generator(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    temperature = 1.0
    max_tokens = 50
    # Use the cropped images for reference.
    system_prompt = """
    Step 1: Identify the features of the object inside the bounding box for each tag number. Use the cropped images for reference. If there are multiple objects of the same category, list each object with its unique color separately.
     1. Color: What is the primary color of the <object>?
     2. Shape: What shape or structural features does the <object> have?
     3. Distinguishing Features: What features most effectively distinguish this <object> from similar objects?
    Example:
     <object>: 1. brown color, 2. rectangle shape, 3. brown color, with a handle 

    Step 2: Create a descriptive tag for each <object> based on your answers. If multiple objects are identified, create separate tags for each.
    Use the format: tag_<tag_number> = []
    Example:
     tag_<tag_number> = ['green <object>', 'rectangular shape <object>', 'green <object> with a handle']
     tag_<tag_number> = ['brown <object>', 'rectangular shape <object>', 'brown <object> with a handle']
    """
#     system_prompt = """
#
# Step 1: What are the main features that distinguish this object from other objects in the same class <object>?
# - Observe the object and define its basic class.
#
# Step 2: Writing a description for each feature
# - For each feature identified in Step 1, provide a detailed description.
# - Explain how that feature helps to distinguish the object from other similar objects.
#
# Step 3: - Write the simplest possible phrase to describe the object. (e.g., "wooden chair")
# - Then, create a slightly more detailed class name to describe the object. (e.g., "black wooden chair")
# - Finally, write the most specific class name you can create for the object. (e.g., "black wooden armchair")
#
# Step 4: extract the description in following format
#  tag_<tag_number> = ['brown <object>', 'rectangular shape <object>', 'brown <object> with a handle']
# """
