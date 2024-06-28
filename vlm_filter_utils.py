from params_proto import PrefixProto, Proto


class vision_filter(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    system_prompt = """
    Objective: Filter out incorrect tags in the image.

Inputs:
Image with Bounding Boxes: (Attach the image with bounding boxes and tag numbers)
List of Tags and Tag Numbers:
tags = ['television', 'white board', 'toy car', 'table', 'trash can']
tag_numbers = [4, 3, 1, 2, 8]

Task: Verify each tag matches the object in its bounding box. If multiple tags of the same object type exist, keep the most accurate one and remove the others.

Output Format:
error = ['incorrect_tag1', 'incorrect_tag2', ...], tag_numbers = ['tag number of incorrect_tag1', 'tag number of incorrect_tag2', ...]

Example Output:
If only one television is in the image but two tags exist, filter out the less accurate tag:
error = ['television'], tag_numbers = ['4']
"""
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
    #system_prompt = """
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