model = load_model("emotion_model__.keras", compile=False)

image1 = cv2.cvtColor(cv2.imread('sad.jpg'), cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(cv2.imread('happy.jpg'), cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

axes[0].imshow(image1)
axes[1].imshow(image2)

plt.tight_layout()
plt.show()

# Prétraitement pour l'IA
# On convertit en gris et on redimensionne à 48x48
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (48, 48))
normalized = resized / 255.0
reshaped = np.reshape(normalized, (1, 48, 48, 1))

# 3. Prédiction
pred = model.predict(reshaped)
print("Probabilités par classe:")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
for i, (emotion, prob) in enumerate(zip(emotions, pred[0])):
    print(f"  {i} - {emotion}: {prob*100:.2f}%")
