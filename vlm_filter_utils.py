from params_proto import PrefixProto, Proto


class vision_filter(PrefixProto):
    SEED: int = 123
    model: str = "gpt-4o-2024-05-13" #"gpt-4o"
    map_google_API: Proto = Proto(env="$MAP_GOOGLE_API_KEY", dtype=str)
    database_dir = None
    temperature = 1.0
    max_tokens = 700
    # Use the cropped images for reference.
    system_prompt = """
   You are an assistant that identifies incorrect tags. You respond according to the given steps. Use the cropped images for reference.
   Step 1. Verify that each tag matches the object in its bounding box. 
       example 1:
       Tag 1 (bag): Incorrect. It is empty
       Tag 4 (apple): Incorrect. It contains <object name>
       Tag 5 (apple): Correct. It is green <object> 
       Tag 6 (soccer ball): Correct. It is red <object> 
       Tag 7 (ball): Correct. It is red <object>
       Tag 8 (ball): Correct. It is red <object>
       Tag 11 (chair): Incorrect. It contains <object name>  
       
   Step 2. Determine if there are multiple tags pointing to the same object among the tags identified as correct in Step 1. Return Tags [number of multiple tags]. If there are no multiple tags for one object, return "no multiple tag".
       example 1:
       Tags[6, 7, 8] : ball, soccer ball, ball are visually pointing to the same object, which is a blue ball under the desk . <describe the look of an object>
   
   Step 3. If there are multiple tags for one object from the response of Step 2, identify which tag is the most accurate. Choose one most precise tag. 
       example 1:
       Tag [6]: object name "soccer ball" is more precise. So, precise_tag = [7]

   Step 4. Provide lists, <empty | incorrect | corrected | duplicated | precise_tags_in_duplicated>_tags =[], results from Steps 1 and 3. [] if No tag.
       example 1:
       empty_tags = [1]
       incorrect_tags = [4, 11] 
       corrected_tags = ['<object name>', '<object name>']
       duplicated_tags = [(6, 7, 8), ()]
       precise_tags_in_duplicated = [7] 
   """
