import json


def get_context(file_path, destination_path):
    """ gets all contexts to be translated from a json file

    Args:
        file_path: The path to the Squad data set to be parsed
        destination_path: The path to the destination file containing all contexts to be translated

    Returns:

    """
    output = {"context": []}
    f = open(file_path, )
    data = json.load(f)
    for i in range(len(data["data"])):
        paragraph = data["data"][i]["paragraphs"]
        for j in range(len(paragraph)):
            context = paragraph[j]["context"]
            output["context"].append(context)

    with open(destination_path, 'w') as outfile:
        json.dump(output, outfile)