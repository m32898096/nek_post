"""Support ``python -m nek_post.cli`` using the console entrypoint."""

from . import main


if __name__ == "__main__":
    raise SystemExit(main())
