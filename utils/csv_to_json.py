import json
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("classes.csv")
    classes = df["class"].tolist()
    json.dump(classes, open("classes.json", "w"))