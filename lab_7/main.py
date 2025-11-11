import numpy as np
from skimage import color



# --- параметры ---
N = 500
np.random.seed(42)


def rgb_to_lab_hue(r, g, b):
		# RGB [0,1] → LAB hue (градусы 0–360)
		rgb = np.array([[[r, g, b]]])   # потому что изображение в библиотеке skimage — это трёхмерный тензор
		print(rgb)
		lab = color.rgb2lab(rgb)
		a, b_ = lab[0, 0, 1], lab[0, 0, 2]
		h = np.degrees(np.arctan2(b_, a)) % 360
		return h


def label_by_hue(h):
	# 1 = тёплый, 0 = холодный, None = нейтральный
	if h <= 90 or h >= 330:
			return 1
	elif 150 <= h <= 270:
			return 0
	else:
			return None

def main():
	data_list = []
	target_list = []

	for _ in range(N):
		r, g, b = np.random.randint(0, 256, 3)
		h = rgb_to_lab_hue(r, g, b)
		label = label_by_hue(h)
		if label is not None:
			data_list.append([r, g, b])
			target_list.append(label)

	# --- готовые массивы ---
	data = np.array(data_list, dtype=np.float32)                      # входы
	target = np.array(target_list, dtype=np.float32).reshape(-1, 1)   # метки

	print(f"Размер data: {data.shape}, размер target: {target.shape}")

	np.savez("color_dataset.npz", data=data, target=target)
	print("Сохранено в color_dataset.npz")



if __name__ == "__main__":
    main()



# Что такое LAB:
# L — Lightness (яркость), от 0 (чёрный) до 100 (белый)
# a — ось от зелёного (–a) к красному (+a)
# b — ось от синего (–b) к жёлтому (+b)