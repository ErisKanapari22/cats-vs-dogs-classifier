from src.predict import predict
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")

    else:
        predict(sys.argv[1])