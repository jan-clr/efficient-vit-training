from datasets import build_transform
from main import get_args_parser
import torch
from PIL import Image


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    transforms = build_transform(True, args)
    print('transforms:', transforms)

    print(transforms.transforms[2])


    # Check if the transforms are correctly built
    # Create a sample PIL input image and apply the transforms
    sample_input = Image.new('RGB', (args.input_size, args.input_size), color='blue')

    try:
        transformed_output = transforms(sample_input)
        print('Transformed output shape:', transformed_output.shape)
    except Exception as e:
        print('Error during transformation:', e)


if __name__ == '__main__':
    main()