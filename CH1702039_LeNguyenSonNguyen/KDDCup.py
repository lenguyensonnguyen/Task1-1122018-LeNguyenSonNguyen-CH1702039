import arff
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

data = arff.load(open("KDDCup99.arff", "r"))

data_raw = data['data']
attributes = data['attributes']

a1 = attributes[1][1]
a2 = attributes[2][1]
a3 = attributes[3][1]

attributes_a = attributes[0:41]
attributes_c = attributes[41:]

data_raw_a = []
data_raw_c = []

data_attributes = []
data_attributes_class = []

le = preprocessing.LabelEncoder()

a_temp = a1 + a2 + a3

le.fit(a_temp)

for item in data_raw:
    item_0 = item[0:1]
    item_13 = item[1:4]
    item_441 = item[4:41]

    item_13 = list(le.transform(item_13))

    item_a = item_0 + item_13 + item_441
    item_b = item[41:]
    data_raw_a.append(item_a)
    data_raw_c.append(item_b)

for att_a in attributes_a:
    data_attributes.append(att_a[0])

for att_c in attributes_c[0][1]:
    data_attributes_class.append(att_c)

X_train, X_test, Y_train, Y_test = train_test_split(data_raw_a,
                                                    data_raw_c,
                                                    test_size=0.33,
                                                    random_state=100)

clf = tree.DecisionTreeClassifier()

model = clf.fit(X_train, Y_train)

data_dot = tree.export_graphviz(model
                                , out_file=None,
                                feature_names=data_attributes,
                                class_names=data_attributes_class,
                                filled=True,
                                rounded=True,
                                special_characters=True
                                )
graph = graphviz.Source(data_dot)

graph.render("PDF_KDDCup99")
print(model)

Y_predict = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_predict) * 100

print("Accuracy: " + accuracy.__str__())
