from params_proto import PrefixProto, Proto


class vision_filter(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None

    system_prompt = """
    Please review the given image and the list of objects. Identify and filter out any tags that are not present in the image or are incorrect. Return the results in the following format:
    error = ['incorrect_tag1', 'incorrect_tag2', ...]
    """
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