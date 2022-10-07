import os


def get_path(*sub_dir):
    return os.path.join(os.path.dirname(__file__), *sub_dir)


def get_parent_path():
    return os.path.dirname(get_path())


def get_data_path(*sub_dir):
    return os.path.join(get_path('data'), *sub_dir)


def get_tests_path(*sub_dir):
    return os.path.join(get_path('tests'), *sub_dir)


def get_tests_data_path(*sub_dir):
    return os.path.join(get_path('tests', 'data'), *sub_dir)
