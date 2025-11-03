from utils.preprocessing import load_dataset, split_data, build_preprocessor
from utils.evaluation import evaluate_and_report, plot_confusion, plot_roc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


if __name__ == "__main__":
    X, y, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(feature_names)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ])

    model.fit(X_train, y_train)

    metrics = evaluate_and_report(model, X_test, y_test)

    out_dir = "outputs/knn"
    plot_confusion(model, X_test, y_test, labels=["malignant", "benign"], title="KNN Confusion", out_dir=out_dir, filename="confusion.png")
    plot_roc(model, X_test, y_test, title="KNN ROC", out_dir=out_dir, filename="roc.png")

    print("Saved plots to:", out_dir)
