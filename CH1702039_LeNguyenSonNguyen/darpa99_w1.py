import arff
import graphviz
from sklearn import tree

data = arff.load(open("DARPA99Week1-2.arff", "r"))

data_train = data['data']
attributes = data['attributes']
attributes_a = attributes[0:22]
attributes_c = attributes[22:]

data_train_a = []
data_train_c = []

data_attributes = []
data_attributes_class = []

for item in data_train:
    item_a = item[0:22]
    item_b = item[22:]

    data_train_a.append(item_a)
    data_train_c.append(item_b)

for att_a in attributes_a:
    data_attributes.append(att_a[0] + "-" + att_a[1])

for att_c in attributes_c[0][1]:
    data_attributes_class.append(att_c)

clf = tree.DecisionTreeClassifier()

model = clf.fit(data_train_a, data_train_c)

data_dot = tree.export_graphviz(model
                                , out_file=None,
                                feature_names=data_attributes,
                                class_names=data_attributes_class,
                                filled=True,
                                rounded=True,
                                special_characters=True
                                )

graph = graphviz.Source(data_dot)

graph.render("PDF_darpa99w1")

print(model)
