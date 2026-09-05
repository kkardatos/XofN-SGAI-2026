import os


def get_clus_path():
    return get_project_path() + "/clus/ClusProject/target/clus-2.12.8-deps.jar"

def get_project_path():
    if os.name == "nt":
        projects_path = "C:\\Users\\Voror\\Projects\\Personal"
    elif os.name == "posix":
        projects_path = "/Users/konkardatos/Projects"
    return projects_path
