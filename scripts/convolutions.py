from skimage.exposure import rescale_intensity
import numpy as np
import cv2
from config import image_path


def convolve(image, kernel):
    # Get the dimensions of the image and kernel
    (img_h, img_w) = image.shape[:2]
    (ker_h, ker_w) = kernel.shape[:2]

    # Allocate memory of the output image, ensuring to 'pad' the borders of the input image so the spacial size
    # (i.e. width and height) are not reduced
    pad = (ker_w - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((img_h, img_w), dtype='float')

    # Loop over the input image, 'sliding' the kernel across each (x, y) coordinate from left-to-right and
    # top-to-bottom
    for y in np.arange(pad, img_h + pad):
        for x in np.arange(pad, img_w + pad):
            # Extract the ROI of the image by extracting the 'center' region of the current (x, y) coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # Perform the actual convolution
            k = (roi * kernel).sum()

            # Store the convolved value in the (x, y) coordinate of the output image
            output[y - pad, x - pad] = k

    # Rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype('uint8')

    # Return the output image
    return output


# average blurring kernel
small_blur = np.ones((7, 7), dtype='float') * (1.0 / (7 * 7))
large_blur = np.ones((21, 21), dtype='float') * (1.0 / (21 * 21))

# sharpening kernel
sharpen = np.array((
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
), dtype='int')

# laplacian edge detection
laplacian = np.array((
    [0,1,0],
    [1,-4,1],
    [0,1,0]
), dtype='int')

kernel_bank = (
    ("small_blur", small_blur),
    ("large_blur", large_blur),
    ("sharpen", sharpen),
    ("laplacian", laplacian)
)

# Read the image
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over each kernel to visualize
for (kernel_name,kernel) in kernel_bank:
    print("[INFO] Applying {} kernel".format(kernel_name))
    convoluted_image = convolve(gray_image, kernel)
    convoluted_image_cv2 = cv2.filter2D(gray_image, -1, kernel)

    # display
    cv2.imshow("Original", gray_image)
    cv2.imshow("{} - convole".format(kernel_name), convoluted_image)
    cv2.imshow("{} - opencv".format(kernel_name), convoluted_image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


