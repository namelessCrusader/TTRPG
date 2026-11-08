# The prototypes here are scripts, not tests (README.md). Some of them run a
# world, or call sys.exit, when they are imported, so pytest must not collect
# them.
collect_ignore_glob = ["*"]
