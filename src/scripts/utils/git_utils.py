import subprocess


def get_git_hash() -> str:
    """
    Returns the current git commit hash.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"