import numpy as np

data = np.load("/home/nathan/IsaacGym_Preview_4_Package/PBHC/example/motion_data/video_motion/processed_data/messi_cel2.npz")
print("Keys:", list(data.keys()))
print("betas shape:", data["betas"].shape)  # Should be (10,)
print("poses shape:", data["poses"].shape)  # Should be (frames, 72)
print("trans shape:", data["trans"].shape)  # Should be (frames, 3)
print("gender:", data["gender"])  # Should be string
