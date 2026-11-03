from housing.data import prepare_data
from housing.preprocessing import build_preprocessor, identify_feature_types

data = prepare_data()

X_train = data["X_train_clean"]

numeric_features, categorical_features = identify_feature_types(X_train)

print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

preprocessor = build_preprocessor(X_train)

X_train_processed = preprocessor.fit_transform(X_train)

print("Processed shape:", X_train_processed.shape)
