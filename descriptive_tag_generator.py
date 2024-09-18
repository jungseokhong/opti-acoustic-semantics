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
Step 1: Identify the features of the object inside the bounding box for each tag number. If there are multiple objects of the same category in the scene, list each object with its unique color separately. Use the cropped images for reference.
 1. Color: What is the primary color of the <object>? If there are multiple objects of the same category, list each object with its unique color separately.
 2. Shape: What shape or structural features does the <object> have?
 3. Unique features: What features most effectively distinguish this <object> from similar objects?
Example:
 <object>: 
 Tag 0 1. green, 2. rectangle shape 3. green, with dot pattern
 tag 0 1. brown color, 2. rectangle shape, 3. brown color, with a handle 
  

Step 2: Create a descriptive tag for each <object> based on your answers. If multiple objects are identified, create seperate tags for each and repeat <tag_0>.
Use the format: tag_<tag_number> = [].
Example:
 tag_0 = ['green <object>', 'rectangular shape <object>', 'green <object> with a handle']
 tag_0 = ['brown <object>', 'rectangular shape <object>', 'brown <object> with a handle']
 tag_<tag_number> = [...]
"""
#Step 1: What are the main features that distinguish this object from other objects in the same class <object>?
#     system_prompt = """
# Step 1: Identify the features of the object inside the bounding box for each tag number. If there are multiple objects of the same category, list each object with its unique features separately.
#
#   <chair> typical characteristics : shape (four legs or wheels), material (plastic, metal, leather)
#           unique features - yellow, light brown Cushion on the sit
#   <chair> unique features - blue, dot pattern on the back
#
# Step 2: Write a short description with <object> and the most specific class name.
# example: yellow <object>, yellow <object> with brown dot on the surface
# example: dot pattern <object>, blue <object> with
#
# Step 3: extract the description in following format
#  tag_<tag_number> = ['ripe <object>', 'ripe <object> with brown dot on the surface']
# """

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
