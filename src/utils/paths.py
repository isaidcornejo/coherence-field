import os


def get_root_dir():
    """
    Return the absolute path to the project root directory.

    The project structure is assumed to be:

        project/
            src/
            paper/
            results/
            ...

    We climb 3 levels from this file:

        src/utils/paths.py → utils → src → project
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return root.replace("\\", "/") 


def get_fig_dirs():
    """
    Return the two figure output directories used in publications:

        paper/revtex/figures/generated
        paper/mdpi/figures/generated

    Returns
    -------
    list[str]
        Absolute paths to both directories.
    """
    root = get_root_dir()

    return [
        os.path.join(root, "paper", "revtex", "figures", "generated").replace("\\", "/"),
        os.path.join(root, "paper", "mdpi", "figures", "generated").replace("\\", "/"),
    ]


def get_results_dir():
    """
    Return the directory used to store NPZ experiment result files.

    Returns
    -------
    str
        Absolute path to results directory.
    """
    return os.path.join(get_root_dir(), "results").replace("\\", "/")
