import torch
import sys
import os

# Add current directory to path so we can import nodes
sys.path.append(os.getcwd())

from nodes import BurveImageCombiner

def test_burve_image_combiner():
    combiner = BurveImageCombiner()
    
    # Create mock images [Batch, Height, Width, Channels]
    # Image 1: Batch size 1
    img1 = torch.rand((1, 64, 64, 3))
    
    # Image 2: Batch size 2
    img2 = torch.rand((2, 64, 64, 3))
    
    print("Testing with two inputs...")
    result = combiner.combine_images(image1=img1, image2=img2)
    output_image = result[0]
    
    print(f"Input 1 shape: {img1.shape}")
    print(f"Input 2 shape: {img2.shape}")
    print(f"Output shape: {output_image.shape}")
    
    expected_shape = (3, 64, 64, 3)
    if output_image.shape == expected_shape:
        print("SUCCESS: Output shape matches expected shape.")
    else:
        print(f"FAILURE: Expected {expected_shape}, got {output_image.shape}")
        
    print("\nTesting with only image1...")
    result = combiner.combine_images(image1=img1, image2=None)
    output_image = result[0]
    print(f"Output shape: {output_image.shape}")
    if output_image.shape == img1.shape:
        print("SUCCESS: Output shape matches input 1.")
    else:
        print("FAILURE")

    print("\nTesting with only image2...")
    result = combiner.combine_images(image1=None, image2=img2)
    output_image = result[0]
    print(f"Output shape: {output_image.shape}")
    if output_image.shape == img2.shape:
        print("SUCCESS: Output shape matches input 2.")
    else:
        print("FAILURE")

    print("\nTesting with no inputs...")
    result = combiner.combine_images(image1=None, image2=None)
    output_image = result[0]
    print(f"Output shape: {output_image.shape}")
    # We expect a default blank image (1, 64, 64, 3)
    if output_image.shape == (1, 64, 64, 3):
        print("SUCCESS: Output shape is default blank image.")
    else:
        print("FAILURE")

    print("\nTesting with mismatched inputs (should resize)...")
    # Image 1: 1024x1024
    img1_large = torch.rand((1, 1024, 1024, 3))
    # Image 2: 928x1232 (as reported by user)
    img2_mismatch = torch.rand((1, 1232, 928, 3))
    
    result = combiner.combine_images(image1=img1_large, image2=img2_mismatch)
    output_image = result[0]
    print(f"Input 1 shape: {img1_large.shape}")
    print(f"Input 2 shape: {img2_mismatch.shape}")
    print(f"Output shape: {output_image.shape}")
    
    expected_shape = (2, 1024, 1024, 3)
    if output_image.shape == expected_shape:
        print("SUCCESS: Output shape matches expected shape (resized).")
    else:
        print(f"FAILURE: Expected {expected_shape}, got {output_image.shape}")

if __name__ == "__main__":
    test_burve_image_combiner()
