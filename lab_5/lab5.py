from PIL import ImageTk
from PIL.Image import Image
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(data='data.yaml', epochs=1)
result = model.predict("image.jpg")  # предсказать на фотографии image.jpg
im = ImageTk.PhotoImage(result.imgsz[..., ::-1])  # преобразовать результат в PIL ImageTk
im.show()  # показать результат на экране
im.save("result.jpg")  # сохранить результат на диск
