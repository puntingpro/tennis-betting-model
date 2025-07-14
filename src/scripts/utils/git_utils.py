import subprocess

def get_git_hash() -> str:
    """
    Retrieves the current git commit hash of the repository.

    Returns:
        str: The full git commit hash as a string, or 'unknown' if it cannot be retrieved.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"