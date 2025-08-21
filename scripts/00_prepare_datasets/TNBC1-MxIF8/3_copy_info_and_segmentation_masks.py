import os
import argparse
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description="Copy segmentation masks and channels.txt for TNBC1-MxIF8 dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing the files to be copied.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory where the files will be copied.')
    return parser.parse_args()

def copy_segmentation_masks(input_dir, output_dir):
    segmentation_input_dir = os.path.join(input_dir, "TME-Analyzer", "segmentations")
    segmentation_output_dir = os.path.join(output_dir, "TME-A_ML6", "segmentations")
    
    if not os.path.exists(segmentation_output_dir):
        os.makedirs(segmentation_output_dir)
    
    for file_name in os.listdir(segmentation_input_dir):
        if file_name.endswith("_cell_segmentation.npz"):
            src_file = os.path.join(segmentation_input_dir, file_name)
            dest_file_name = file_name.replace("_cell_segmentation", "")
            dest_file = os.path.join(segmentation_output_dir, dest_file_name)
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

def copy_channels_file(input_dir, output_dir):
    src_file = os.path.join(input_dir, "channels.txt")
    dest_file = os.path.join(output_dir, "channels.txt")
    shutil.copy(src_file, dest_file)
    print(f"Copied {src_file} to {dest_file}")

def main():
    args = parse_arguments()
    
    # Copy segmentation masks
    copy_segmentation_masks(args.input_dir, args.output_dir)
    
    # Copy channels.txt
    copy_channels_file(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
