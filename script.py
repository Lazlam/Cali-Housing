import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from zipfile import ZipFile
import os
import tensorflow as tf
import keras


zip_file = "housing.csv.zip"


with ZipFile(zip_file, "r") as z:
    z.extractall("")
    csv_filename = [name for name in z.namelist() if name.endswith(".csv")][0]

data_path = f"{csv_filename}"


if os.path.exists(data_path):
    print(f"Το αρχείο αποσυμπιέστηκε με επιτυχία: {data_path}")
else:
    print("Πρόβλημα στην αποσυμπίεση του αρχείου.")


df = pd.read_csv(data_path)


print(df.head())



print(df.isnull().sum())


num_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                'population', 'households', 'median_income']
cat_features = ['ocean_proximity']


imputer = SimpleImputer(strategy='median')
df[num_features] = imputer.fit_transform(df[num_features])


ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe_encoded = ohe.fit_transform(df[cat_features])
ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(cat_features))


final_df = pd.concat([df[num_features], ohe_df, df['median_house_value']], axis=1)


scaler = StandardScaler()
final_df[num_features] = scaler.fit_transform(final_df[num_features])


final_df.to_csv('processed_housing.csv', index=False)


print("Προεπεξεργασία ολοκληρώθηκε!")


plt.figure(figsize=(12, 8))
sns.histplot(df['median_house_value'], bins=50, kde=True)
plt.title("Κατανομή των Τιμών των Ακινήτων")
plt.xlabel("Τιμή Ακινήτου")
plt.ylabel("Συχνότητα")
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['median_income'], y=df['median_house_value'], alpha=0.5)
plt.title("Σχέση Μεταξύ Εισοδήματος και Τιμής Ακινήτου")
plt.xlabel("Διάμεσο Εισόδημα")
plt.ylabel("Διάμεση Τιμή Ακινήτου")
plt.show()


df_final = pd.concat([df[num_features], ohe_df], axis=1)

plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Χάρτης Θερμότητας των Χαρακτηριστικών")
plt.show()


threshold = df['median_house_value'].median()


y = np.where(df['median_house_value'] > threshold, 1, -1)


X = df[['median_income', 'households']].values


X = np.c_[np.ones(X.shape[0]), X]


class Perceptron:
    def __init__(self, learning_rate=0.1, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None

    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])


        for _ in range(self.n_iter):
            for xi, target in zip(X, y):

                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi

    def predict(self, X):

        return np.where(np.dot(X, self.weights) >= 0, 1, -1)


perceptron = Perceptron(learning_rate=0.1, n_iter=1000)
perceptron.fit(X, y)


predictions = perceptron.predict(X)


accuracy = np.mean(predictions == y)
print(f"Ακρίβεια του μοντέλου: {accuracy * 100:.2f}%")


plt.figure(figsize=(8, 6))
plt.scatter(df['median_income'], df['households'], c=y, cmap='bwr', marker='o', label='Δεδομένα')


x_min, x_max = df['median_income'].min() - 1, df['median_income'].max() + 1
y_min, y_max = df['households'].min() - 1, df['households'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = perceptron.predict(np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title("Γραμμική Διάκριση με Perceptron")
plt.xlabel("Μέσο Εισόδημα")
plt.ylabel("Αριθμός Νοικοκυριών")
plt.show()


threshold = df['median_house_value'].median()
y = np.where(df['median_house_value'] > threshold, 1, -1)


X = df[['median_income', 'households']].values


X = np.c_[np.ones(X.shape[0]), X]


class LeastSquares:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):

        X_transpose = X.T
        self.weights = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    def predict(self, X):

        return X.dot(self.weights)


ls_model = LeastSquares()
ls_model.fit(X, y)


predictions = ls_model.predict(X)


accuracy = np.mean(np.sign(predictions) == y)
print(f"Ακρίβεια του μοντέλου Least Squares: {accuracy * 100:.2f}%")


plt.figure(figsize=(8, 6))
plt.scatter(df['median_income'], df['households'], c=y, cmap='bwr', marker='o', label='Δεδομένα')


x_min, x_max = df['median_income'].min() - 1, df['median_income'].max() + 1
y_min, y_max = df['households'].min() - 1, df['households'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))


Z = ls_model.predict(np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title("Γραμμική Παλινδρόμηση με Least Squares")
plt.xlabel("Μέσο Εισόδημα")
plt.ylabel("Αριθμός Νοικοκυριών")
plt.show()





X = final_df.drop(columns=['median_house_value']).values
y = final_df['median_house_value'].values


y = y / np.max(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)


test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Εξέλιξη της Απόδοσης του Νευρωνικού Δικτύου')
plt.show()


y_pred = model.predict(X_test)


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel('Πραγματική Τιμή Ακινήτου')
plt.ylabel('Προβλεπόμενη Τιμή Ακινήτου')
plt.title('Πραγματικές vs Προβλεπόμενες Τιμές')
plt.show()
