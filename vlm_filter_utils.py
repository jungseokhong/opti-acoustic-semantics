

from params_proto import PrefixProto, Proto


class vision_filter(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    system_prompt = """
    You are an assistant that identifies incorrect tags. You respond according to the given steps.
    Step 1. Verify that each tag matches the object in its bounding box.
        example 1:
        Tag 1 (bag): Correct.
        Tag 4 (apple): Incorrect. It is a [object name in the bounding box]
    Step 2. Determine if there are multiple tags pointing to the same object. If there are no multiple tags for one object, return "no multiple tag".
        example 1:
        Tags [number of multiple tags] or No multiple tag

    Step 3. If there are multiple tags for one object from the response of Step 2, visually identify which tag most accurately covers the entire object. Rank the tags and explain your reasoning.
        example 1:
        Tags [the number of multiple tags from Step 2]: [explain your reasoning]

    Step 4. Provide the conclusions of Step1 and Step 3, in the format: unmatched_tags = [ tag number, tag number,...]. Return unmatched_tags = [] if No unmatched tags. 

    Step 5. Extract only the list, unmatched_tags = [ tag number, tag number, ... ] from the response of Step 4.   
    """

class vision_another_filter(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    system_prompt = """
    You are an assistant that identifies incorrect tags. When text tags do not match the given scene or when multiple tags are assigned to a single object, you determine the most accurate tag and identify the others as incorrect tags. You respond according to the given steps.
    Step 1. Do you agree with the unmatched_tags?
    Step 2. Do you agree with the unmatched_tags?The given tags might be pointing to the same object. which one is more precise in covering the overall object? rank them and explain. 
      - output examples 1:
        - television = [3, 4, 7] 
        - Tag 3,4, and 7 are pointing to one object. Baseball hat is more precise tag than hat since there is LA mark on it.  Considering the spatial relationships between the remaining tags, Tag 3 Focuses on a smaller part of the baseball hat, not covering the entire object. Tag 4 The most precise one.
    Step 3. Provide the conclusions of Step1 and Step2, in the format: unmatched_tags = [ tag number, tag number,...]. Return unmatched_tags = [] if No unmatched tags. 
      - output examples 1: 
        - Step1: unmatched_tags = [0]
        - Step2: multi_tags = [3, 7]
    Step 4. Extract only the list, unmatched_tags = [ tag number, tag number, ... ] from the response of Step 3.
      - output examples 1: incorrect_tags = [0, 3, 7]     
    """

# bbox filstering
# mode - > Step

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
