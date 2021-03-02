import os


def get_folder_content(path, everything=False):
    """
    Get the names of all files and directories from the given path

    :param path: where the content is located on disk

    :return: (files, directories) - a tuple with an array of files and an array of directories
    """

    files = []
    directories = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        # files.extend(file_names)

        for file in file_names:
            files.append(dir_path+"\\"+file)

        directories.extend(dir_names)
        if not everything:
            break

    return files, directories