from dedalus.tools import post
post.merge_process_files("snapshots", cleanup=True)
post.merge_process_files("profiles", cleanup=True)
post.merge_process_files("series", cleanup=True)

