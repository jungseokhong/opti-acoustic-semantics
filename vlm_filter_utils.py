

from params_proto import PrefixProto, Proto


class vision_filter(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    temperature = 1.0
    max_tokens = 700
    system_prompt = """
   You are an assistant that identifies incorrect tags. You respond according to the given steps.
   Step 1. Verify that each tag matches the object in its bounding box. Use the cropped images for reference.
       example 1:
       Tag 1 (bag): Incorrect. It is empty
       Tag 4 (apple): Incorrect. It contains <object name>
       Tag 5 (apple): Correct 
       Tag 6 (soccer ball): Correct
       Tag 7 (ball): Correct
       Tag 11 (chair): Incorrect. It contains <object name>  
       
   Step 2. Determine if there are multiple tags pointing to the same object among the tags identified as correct in Step 1. Return Tags [number of multiple tags]. If there are no multiple tags for one object, return "no multiple tag".
       example 1:
       Tags[6, 7] : ball, soccer ball are visually pointing to the same object, which is a blue ball under the desk . <describe the look of an object>
   
   Step 3. If there are multiple tags for one object from the response of Step 2, identify which tag is the most accurate. Choose one most precise tag. 
       example 1:
       Tag [6]: object name "soccer ball" is more precise. So, precise_tag = [7]

   Step 4. Provide lists, <empty | incorrect | corrected | duplicated>_tags =[], results from Steps 1 and 3. [] if No tag.
       example 1:
       empty_tags = [1]
       incorrect_tags = [4, 11] 
       corrected_tags = ['<object name>', '<object name>']
       duplicated_tags = [6] 
   """

#     system_prompt = """
# You are an assistant to tag objects with classes.
#
# 1. List the objects in the scene.
#   example 1:
#   tag = [class name, class name, ...]
# """
#
# Image Tagging and Object Recognition Prompt
# Given an image with bounding boxes, the cropped images corresponding to each bounding box, and the tags, perform the following tasks:
# 1. Remove Background Tags:
# - Delete tags that represent Background elements from the given tags
# - Example 1:
#   remove=[box number, ...]
# 3. Describe Objects Based on Step 2 and The first image has bounding boxes:
# - Generate a detailed description.
# - Example 1:
#   [box number] has [shape], [color], [main material] and [distinctive feature] on [location] and is used for [function]
#
# 4. Class and Tag Comparison:
# - For each bounding box, determine whether the identified object class matches the given main tag category.
# - Example 1:
#   [box number]: it is [class name] is ['correct', 'incorrect', or 'not_present']. the tag is [the given tag].
#
# 5. Description and Tag comparison:
# - Check if the generated description matches the given tag category.
# - Example 1:
#   [box number]: [the given tag] is ['correct', 'incorrect', or 'not_present'] to [reasoning, the description that matches the given tag]
#
# 6. Identify Multiple Tags:
# - Determine if multiple tags exist for the same object.
# - Example 1:
#   [object name or object names]: [box numbers].
# - Example 2:
#   [hat] : [1,3]
#   [Table, desk] : [4, 6] have different tag names but similar category pointing one object visually.
#   [cup, coffee cup] : [box number, ...]
#
# 7. Select the Best box from Multiple Tags:
# - Choose the tag that sufficiently covers the object while including fewer surrounding elements.
# - Example 1: [the Best box number] is the most accurate box. remove rest of [box numbers]
#
# 8. Update Tags:
# Based on only step 1, 3 and 7 results:
# - Keep the tag if correct
# - Provide the identified object name if incorrect.
# - Remove the tag if empty, represents a background element, or is a less accurate duplicate
# - example 1:
#   keep = []
#   revise = [tag number: newclass name, ...]
#   remove = [tag numbers, ...]


    # system_prompt = """
    #  You are an assistant that identifies incorrect tags. Follow these steps:
    # Step 1: Each box is labeled with a number and color, indicating the location of an object. Use the cropped images corresponding to each bounding box to identify the visible objects. List the actual visible objects in the format . If the box is empty, write "empty". Ignore the provided tags completely. Only respond based on what is actually visible in the cropped images.
    #   output example 1:
    #     [Box Number: Box Color] [Object Name]
    #
    # Step 2. If the object matches to the given tags, add it to the "matches" list. If it doesn't match, add it to the "correct" list. If the box is empty, add the tag to the "delete" list. Return [] if the list is empty.
    #   output example 1:
    #     matches = [box number, ...]
    #     correct = [box number, ...]
    #     delete = [box number, ...]
    #
    # Step 3. Extract only the lists, matches = [], matches = [], correct = [] from Step 3.
    #   output example 1: matches = [box number, ...], correct = [box number, ...], delete = [box number, ...]
    # """
    # system_prompt = """
    # You are an assistant that identifies incorrect tags. You respond according to the given steps.
    # Step 1. Each box, labeled with a number and color, shows an object's location. List the actually visible objects as [Box Number] [Box Color]: [Object Name]. If empty, write "empty". Ignore the given tags; only respond with what is actually visible.
    #
    # Step 2. If the object matches, add it to the "matches" list. If it doesn't match, add it to the "correct" list. If the box is empty, add the tag to the "delete" list. Return [] if the list is empty.
    #     example 1:
    #     matches = [box number]
    #     correct = []
    #     delete = [box number, box number , ...]
    # """

class vision_another_filter(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    system_prompt = """
    
    """

    # system_prompt = """
    # You are an assistant that identifies incorrect tags. When text tags do not match the given scene or when multiple tags are assigned to a single object, you determine the most accurate tag and identify the others as incorrect tags. You respond according to the given steps.
    # Step 1. Do you agree with the unmatched_tags?
    # Step 2. Do you agree with the unmatched_tags?The given tags might be pointing to the same object. which one is more precise in covering the overall object? rank them and explain.
    #   - output examples 1:
    #     - television = [3, 4, 7]
    #     - Tag 3,4, and 7 are pointing to one object. Baseball hat is more precise tag than hat since there is LA mark on it.  Considering the spatial relationships between the remaining tags, Tag 3 Focuses on a smaller part of the baseball hat, not covering the entire object. Tag 4 The most precise one.
    # Step 3. Provide the conclusions of Step1 and Step2, in the format: unmatched_tags = [ tag number, tag number,...]. Return unmatched_tags = [] if No unmatched tags.
    #   - output examples 1:
    #     - Step1: unmatched_tags = [0]
    #     - Step2: multi_tags = [3, 7]
    # Step 4. Extract only the list, unmatched_tags = [ tag number, tag number, ... ] from the response of Step 3.
    #   - output examples 1: incorrect_tags = [0, 3, 7]
    # """

# bbox filstering
# mode - > Step

    # system_prompt = """
    # You are an assistant that identifies incorrect tags. You respond according to the given steps.
    # Step 1. Verify that each tag matches the object in its bounding box.
    #     example 1:
    #     Tag 1 (bag): Correct.
    #     Tag 4 (apple): Incorrect. It is a [object name in the bounding box]
    #     Tag 6 (ball): Correct
    #     Tag 7 (ball): Correct
    # Step 2. Determine if there are multiple tags pointing to the same object and return Tags [number of multiple tags]. If there are no multiple tags for one object, return "no multiple tag".
    #     example 1:
    #     Tags [6, 7] are pointing to the same object
    #
    # Step 3. If there are multiple tags for one object from the response of Step 2, visually identify which tag most accurately covers the entire object. Rank the tags and explain your reasoning.
    #     example 1:
    #     Tags [the number of multiple tags from Step 2]: [explain your reasoning]. Therefore, precise_tag = [7]
    #
    # Step 4. Provide the conclusions of Step1 and Step 3, in the format: unmatched_tags = [ tag number, tag number,...]. Return unmatched_tags = [] if No unmatched tags.
    #     example 1:
    #     unmatched_tags = [4]
    #     unmatched_tags = [6]
    #
    # Step 5. Extract only the list, unmatched_tags = [ tag number, tag number, ... ] from the response of Step 4.
    #     example 1:
    #     unmatched_tags = [4, 6]
    # """

# filtering  using the object list
#     system_prompt = """
#     Objective: Filter out incorrect tags in the image.
#
# Inputs:
# Image with Bounding Boxes: (Attach the image with bounding boxes and tag numbers)
# List of Tags and Tag Numbers:
# tags = ['television', 'white board', 'toy car', 'table', 'trash can']
# tag_numbers = [4, 3, 1, 2, 8]
#
# Task: Verify each tag matches the object in its bounding box. If multiple tags of the same object type exist, keep the most accurate one and remove the others.
#
# Output Format:
# error = ['incorrect_tag1', 'incorrect_tag2', ...], tag_numbers = ['tag number of incorrect_tag1', 'tag number of incorrect_tag2', ...]
#
# Example Output:
# If only one television is in the image but two tags exist, filter out the less accurate tag:
# error = ['television'], tag_numbers = ['4']
# """
#     system_prompt = """
# Given the following inputs:
#
# 1. Image with Bounding Boxes:
#    (Attach the image with bounding boxes and tag numbers overlayed)
#
# 2. List of Tags and Corresponding Tag Numbers:
#    - tags = ['television', 'white board', 'toy car', 'table', 'trash can']
#    - tag_numbers = [4, 3, 1, 2, 8]
#
# Determine which tags do not correctly correspond to the objects within their bounding box areas. An incorrect tag could be one that does not accurately match the object within the bounding box. If multiple tags of the same object type exist but only one is present in the image, filter out the less accurate tag.
# Return ONLY the results in the following format: error = ['incorrect_tag1', 'incorrect_tag2', ...], tag_numbers = ['tag number of incorrect_tag1', 'tag number of incorrect_tag2', ...]
#     """
# system_prompt = """
# Please review the given image and the list of objects. Identify and filter out any tags that are not present in the image or are incorrect. Return the results in the following format:
# error = ['incorrect_tag1', 'incorrect_tag2', ...]
# """
# system_prompt = """
# a user will be given a list of objects that are supposed to be in a photo. Your task is to determine if all the objects in the list are present in the photo.
#
# Respond according to the following format:
# 1. If there are objects in the list that are not present in the photo:
#    - user: List of objects=['object1', 'object2', ...]
#    - system: Error=['object1', 'object2', ...]
#
# 2. If all objects in the list are present in the photo:
#    - user: List of objects=['object1', 'object2', ...]
#    - system: Error=[]
#
# Example 1:
# - yser: List of objects = ['person', 'sky']
# - system: Error=[]
# Example 2:
# -  yser: List of objects = ['person', 'sky', 'tree']
# - system: Error=[tree]
#
# Now, let's start:
# """
