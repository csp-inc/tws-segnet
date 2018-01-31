from image_tools import file_matcher

input_dir = "./images/train"

# for the aa files
src_d = "%s/ab"%input_dir
to_match = "%s/aa"%input_dir
file_matcher(src_d, to_match)

# for the b files
to_match = "%s/b"%input_dir
file_matcher(src_d, to_match)


