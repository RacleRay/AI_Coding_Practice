import config
from models.model import Model


if __name__ == "__main__":
    model = Model(model_path=config.lgb_path, train_mode=False, debug_mode=False)

    title = "这是书名"
    desc = "这是摘要"

    label = model.predict(title, desc)
    print(f"Predict: {label}\n")