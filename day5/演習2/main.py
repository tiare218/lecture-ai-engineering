import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import time


# ---------------------------
# データ前処理関数
# ---------------------------
def categorize_age(age):
    if pd.isnull(age):
        return "unknown"
    elif age <= 12:
        return "child"
    elif age <= 19:
        return "teen"
    elif age < 60:
        return "adult"
    else:
        return "senior"


def preprocess_titanic_data(df):
    df = df.copy()

    # 特徴量追加
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Title 抽出・変換
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace(["Mme"], "Mrs")
    df["Title"] = df["Title"].replace(
        [
            "Dr",
            "Rev",
            "Col",
            "Major",
            "Capt",
            "Jonkheer",
            "Don",
            "Sir",
            "the Countess",
            "Dona",
        ],
        "Rare",
    )
    title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    df["Title"] = df["Title"].map(title_map).fillna(4)

    # AgeGroup（カテゴリカル特徴量）
    df["AgeGroup"] = df["Age"].apply(categorize_age)
    age_group_map = {"child": 0, "teen": 1, "adult": 2, "senior": 3, "unknown": 4}
    df["AgeGroup"] = df["AgeGroup"].map(age_group_map)

    # 欠損値補完
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    # カテゴリ変数の数値化
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    return df


# ---------------------------
# メイン処理
# ---------------------------

# データ読み込み（旧スタイル: train.csv のみ）
df = pd.read_csv("data/train.csv")

# 訓練・テスト分割
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 前処理
train_df = preprocess_titanic_data(train_df)
test_df = preprocess_titanic_data(test_df)

# 説明変数と目的変数
features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
    "Title",
]
# features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "FarePerPerson", "Title", "AgeGroup"]
X_train = train_df[features]
y_train = train_df["Survived"]
X_test = test_df[features]
y_test = test_df["Survived"]

# ---------------------------
# モデルとハイパーパラメータチューニング
# ---------------------------

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [2, 10, 30],
    "min_samples_split": [2, 10],
}

model = RandomForestClassifier(random_state=42)

start_time = time.time()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

end_time = time.time()

# 最良モデルで推論
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 精度評価
accuracy = accuracy_score(y_test, y_pred)

# ---------------------------
# 結果出力
# ---------------------------
print(f"最良モデルのパラメータ: {grid_search.best_params_}")
print(f"テストセット精度: {accuracy:.4f}")
print(f"チューニングにかかった時間: {end_time - start_time:.2f}秒")
