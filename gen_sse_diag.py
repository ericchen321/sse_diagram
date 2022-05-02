from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# get image stats
img =  np.asarray(Image.open("asian_dragon00.png").convert('L'), dtype=np.uint8)
img_size = 900
img = img[0:img_size, 170:170+img_size]
print(f"image shape: {img.shape}")

# binarize
img_bin = (img > 0.0).astype(np.float32)
plt.imshow(img_bin)
#plt.title("binary")
plt.show()
#pilimg_bin = Image.fromarray(img_bin.astype(np.uint8)).convert("L")
#pilimg_bin.save("dragon_binary.png")

# discretize into neighborhoods:
neighborhood_size = 5
# expand each neighborhood into a 1D vector
neighborhoods = img_bin.reshape(
    int(img_size/neighborhood_size),
    neighborhood_size,
    int(img_size/neighborhood_size),
    neighborhood_size).transpose(0, 2, 1, 3)
print(f"neighborhoods shape: {neighborhoods.shape}")
neighborhoods_flat = neighborhoods.reshape(
    -1,
    neighborhood_size*neighborhood_size)
print(f"neighborhoods_flat shape: {neighborhoods_flat.shape}")
# compute mean
means = np.expand_dims(
    np.mean(neighborhoods_flat, 1), 1)
print(f"means shape: {means.shape}")
# compute SSE
sses = np.diag(np.matmul(
    (neighborhoods_flat - means),
    (neighborhoods_flat - means).transpose(1, 0)
)).reshape(
    int(img_size/neighborhood_size),
    int(img_size/neighborhood_size)
)
print(f"sses shape: {sses.shape}")

# visualize sses as a heatmap
hmap = sses / np.amax(np.abs(sses))
ax = sns.heatmap(hmap)
plt.title(f"{neighborhood_size}x{neighborhood_size} Neighborhood SSE")
plt.savefig("dragon_sse.png")
